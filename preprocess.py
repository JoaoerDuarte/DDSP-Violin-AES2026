"""Preprocess the .wav files of one split, <data.audio_dir>/<split>/ (searched recursively): segments
of model.signal_length samples with their f0 (pYIN), A-weighted loudness and harmonic amplitudes per
frame of model.block_size samples, the node order of each file and its name.

A file name like G_m000.00p_F0.5953N_vb0.6231v_z0.2505x_k4_r003945.wav gives the string, the
semitones above the open string and, after _k, the node order. The pitch search then stays within
7 semitones of the nominal pitch. Other names are searched over 180 to 1400 Hz, with node order 0.
"""
import argparse
import os
import pathlib
import re

import librosa
import numpy as np
import torch
import yaml
from tqdm import tqdm

from ddsp_torch.core import extract_harmonic_amplitudes, extract_loudness, extract_pitch

ARRAYS = ("signals", "pitches", "loudness", "harmonic_amps", "k_class", "filenames")
N_HARMONICS = 40
OPEN_STRINGS = {'G': 196.00, 'D': 293.66, 'A': 440.00, 'E': 659.26}  # open-string f0 in Hz
FILENAME = re.compile(r'(?P<string>[GDAE])_m(?P<semitone>[\d.]+)p_.*_k(?P<k>\d+)_')


def load_config(name: str) -> dict:
    """Load configs/<name>.yaml."""
    with open(os.path.join("configs", f"{name}.yaml")) as f:
        return yaml.safe_load(f)


def data_dirs(config: dict, split: str = "train") -> tuple[str, str]:
    """Folder of the split's .wav files and folder of their preprocessed arrays."""
    audio_dir = config.get("data", {}).get("audio_dir", "dataset/audio")
    name = os.path.basename(os.path.normpath(audio_dir))
    return os.path.join(audio_dir, split), os.path.join("preprocessed", name, split)


def parse_filename(path) -> tuple[float | None, int]:
    """Nominal f0 in Hz (None if the name does not give it) and node order of a file."""
    match = FILENAME.match(os.path.basename(path))
    if match is None:
        return None, 0
    return OPEN_STRINGS[match['string']] * 2.0 ** (float(match['semitone']) / 12.0), int(match['k'])


def preprocess_file(path, sampling_rate: int, block_size: int, signal_length: int):
    """Resample one file and zero-pad it to whole segments. Returns its arrays, one row per segment."""
    audio, _ = librosa.load(path, sr=sampling_rate)
    audio = np.pad(audio, (0, -len(audio) % signal_length))
    target_f0, k = parse_filename(path)
    n_segments, frames = len(audio) // signal_length, signal_length // block_size
    n_frames = n_segments * frames  # pYIN and the loudness STFT give one frame more
    pitch = extract_pitch(audio, sampling_rate, block_size, target_f0)[:n_frames].astype(np.float32)
    loudness = extract_loudness(audio, sampling_rate, block_size)[:n_frames]
    amps = extract_harmonic_amplitudes(audio, pitch, sampling_rate, block_size, N_HARMONICS)
    return (audio.reshape(n_segments, signal_length), pitch.reshape(n_segments, frames),
            loudness.reshape(n_segments, frames), amps.reshape(n_segments, frames, N_HARMONICS),
            np.full(n_segments, k, dtype=np.int64), np.full(n_segments, os.path.basename(path)))


def preprocess(config: dict, split: str = "train"):
    """Preprocess every .wav file of the split and save the arrays."""
    model = config["model"]
    audio_dir, out_dir = data_dirs(config, split)
    files = sorted(pathlib.Path(audio_dir).rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav files in {audio_dir}")
    results = [preprocess_file(f, model["sampling_rate"], model["block_size"], model["signal_length"])
               for f in tqdm(files, desc=f"Preprocessing {audio_dir}")]
    os.makedirs(out_dir, exist_ok=True)
    for name, arrays in zip(ARRAYS, zip(*results)):
        np.save(os.path.join(out_dir, f"{name}.npy"), np.concatenate(arrays))
    print(f"Saved {sum(len(r[0]) for r in results)} segments to {out_dir}")


class Dataset(torch.utils.data.Dataset):
    """Preprocessed segments as (audio, f0, loudness, harmonic amplitudes, node order)."""

    def __init__(self, data_dir: str):
        self.signals, self.pitches, self.loudness, self.harmonic_amps, self.k_class = (
            np.load(os.path.join(data_dir, f"{name}.npy")) for name in ARRAYS[:5])

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, i):
        return (torch.from_numpy(self.signals[i]), torch.from_numpy(self.pitches[i]),
                torch.from_numpy(self.loudness[i]), torch.from_numpy(self.harmonic_amps[i]),
                torch.tensor(int(self.k_class[i]), dtype=torch.long))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="ddsp_violin", help="name of a config in configs/")
    parser.add_argument("--split", default="train", choices=("train", "test"))
    args = parser.parse_args()
    preprocess(load_config(args.config), args.split)
