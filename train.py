"""Train a model, e.g. python train.py --config ddsp_violin or --config baseline.

Preprocesses the train split first if its arrays are missing. Writes config.yaml, loss_log.txt and
final_state.pth to runs/<config>/.
"""
import argparse
import os
import subprocess
import sys

import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from ddsp_torch.model_utils import NODE_ORDERS
from ddsp_torch.train_util import LOUD_DB, calculate_loss, initialize_model, setup_loss_functions
from preprocess import ARRAYS, Dataset, data_dirs, load_config

# Training settings of the papers. A config can override them under "train".
TRAIN_DEFAULTS = {
    "steps": 10000,
    "batch_size": 8,
    "learning_rate": 1e-3,
    "lr_decay_rate": 0.98,  # learning-rate factor per lr_decay_steps, spread over every step
    "lr_decay_steps": 10000,
    "gradient_clip_norm": 3.0,
    "num_workers": 4,
}


def supervised_ce(logits, k_class, loudness):
    """Cross-entropy of the class logits [B, T, C] against the node order of each sample, on loud frames."""
    target = torch.tensor([NODE_ORDERS.index(k) for k in k_class.tolist()], device=logits.device)
    target = target.unsqueeze(1).expand(logits.shape[0], logits.shape[1])
    loud = loudness.squeeze(-1) > LOUD_DB
    return F.cross_entropy(logits[loud], target[loud]) if loud.any() else None


def train(config_name: str):
    config = load_config(config_name)
    settings = {**TRAIN_DEFAULTS, **config.get("train", {})}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, data_dir = data_dirs(config)
    if not all(os.path.exists(os.path.join(data_dir, f"{name}.npy")) for name in ARRAYS):
        subprocess.run([sys.executable, "preprocess.py", "--config", config_name], check=True)
    loader = torch.utils.data.DataLoader(
        Dataset(data_dir), settings["batch_size"], shuffle=True, drop_last=True,
        num_workers=settings["num_workers"], pin_memory=device.type == "cuda",
        persistent_workers=settings["num_workers"] > 0,
    )
    if len(loader) == 0:
        raise ValueError(f"{data_dir} holds fewer segments than one batch")

    model = initialize_model(config, device)
    mssl, hrl, hrl_active = setup_loss_functions(config, device)
    ce_weight = float(config["loss"].get("supervised_ce_weight", 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: settings["lr_decay_rate"] ** ((step + 1) / settings["lr_decay_steps"]))

    run_dir = os.path.join("runs", config_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    step = 0
    model.train()
    with open(os.path.join(run_dir, "loss_log.txt"), "w") as log, \
            tqdm(total=settings["steps"], desc="Training") as progress:
        log.write("step,total_loss,spectral_loss,hrl_loss\n")
        while step < settings["steps"]:
            for audio, pitch, loudness, harmonic_amps, k_class in loader:
                audio = audio.to(device, non_blocking=True)
                pitch = pitch.unsqueeze(-1).to(device, non_blocking=True)
                loudness = loudness.unsqueeze(-1).to(device, non_blocking=True)
                outputs = model(pitch, loudness, audio, harmonic_amps.to(device))
                total, spectral, hrl_value, _ = calculate_loss(mssl, hrl, outputs, audio, pitch, loudness,
                                                                hrl_active)
                if ce_weight > 0:
                    ce = supervised_ce(outputs["violin_diagnostics"]["softmax_k_logits"], k_class, loudness)
                    if ce is not None:
                        total = total + ce_weight * ce
                optimizer.zero_grad()
                total.backward()
                clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"])
                optimizer.step()
                scheduler.step()

                hrl_text = "" if hrl_value is None else f"{hrl_value:.6f}"
                log.write(f"{step},{total.item():.6f},{spectral.item():.6f},{hrl_text}\n")
                progress.set_postfix(loss=f"{total.item():.4f}")
                progress.update()
                step += 1
                if step == settings["steps"]:
                    break
    torch.save(model.state_dict(), os.path.join(run_dir, "final_state.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ddsp_violin", help="name of a config in configs/")
    train(parser.parse_args().config)
