"""Body-filter errors of trained runs against measured body responses.

    python evaluate_filter.py --responses <folder of .wav files> runs/<config> [runs/<config> ...]

Each run is compared with the response whose file name, without .wav, appears in the run
folder's name. The learnt filter is the FIR taps, or the impulse response of an AR or ARMA
filter. Delta is the dB difference between the two magnitude spectra (8192-point FFT) over
100 Hz to 8 kHz. MC-LSD is the RMS of Delta around its mean, the tilt error is the change of
the straight line fitted to Delta across the band, and the resonance error is the RMS of
Delta around that line. Prints each run, then the mean and standard deviation over the runs.
"""
import argparse
import os

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import yaml
from scipy.fft import fft, fftfreq

from ddsp_torch.filters import body_filter

N_FFT, BAND_HZ, DB_OFFSET = 8192, (100.0, 8000.0), 1e-10
FILTER_SETTINGS = ("resonance_type", "resonance_length", "resonance_ar_order", "resonance_ma_order",
                   "resonance_window_size", "resonance_max_reflection")


def magnitude_db(h: np.ndarray, sampling_rate: int):
    """Frequencies and magnitude spectrum (dB) of h over BAND_HZ, h zero-padded or cut to N_FFT."""
    x = np.pad(h, (0, max(0, N_FFT - len(h))))[:N_FFT]
    freqs = fftfreq(N_FFT, 1 / sampling_rate)
    keep = freqs >= 0
    freqs, mag = freqs[keep], np.abs(fft(x, n=N_FFT))[keep]
    band = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    return freqs[band], 20 * np.log10(mag[band] + DB_OFFSET)


def measured_response(path: str, sampling_rate: int) -> np.ndarray:
    """Measured impulse response, mono, resampled to sampling_rate."""
    h, sr = sf.read(path)
    if h.ndim > 1:
        h = h.mean(axis=1)
    if sr != sampling_rate:
        h = scipy.signal.resample(h, int(len(h) * sampling_rate / sr))
    return h


def learnt_response(run_dir: str):
    """Impulse response of the run's trained body filter, and the run's sampling rate."""
    with open(os.path.join(run_dir, "config.yaml")) as f:
        model_config = yaml.safe_load(f)["model"]
    resonance = body_filter(**{k: model_config[k] for k in FILTER_SETTINGS if k in model_config})
    state = torch.load(os.path.join(run_dir, "final_state.pth"), map_location="cpu", weights_only=True)
    resonance.load_state_dict({k[len("resonance."):]: v for k, v in state.items() if k.startswith("resonance.")})
    with torch.no_grad():
        h = resonance.impulse_response()
    return h.detach().numpy().astype(np.float64), model_config["sampling_rate"]


def filter_errors(freqs: np.ndarray, true_db: np.ndarray, learnt_db: np.ndarray):
    """MC-LSD, tilt error and resonance error in dB."""
    delta = true_db - learnt_db
    mc_lsd = np.sqrt(np.mean((delta - delta.mean()) ** 2))
    slope, intercept = np.polyfit(freqs, delta, 1)
    tilt = abs(slope * (freqs[-1] - freqs[0]))
    resonance = np.sqrt(np.mean((delta - (slope * freqs + intercept)) ** 2))
    return mc_lsd, tilt, resonance


def match_response(run_dir: str, responses: list[str]) -> str:
    """The response whose file name (without .wav) is the longest found in the run folder's name."""
    name = os.path.basename(os.path.normpath(run_dir))
    matches = [r for r in responses if os.path.splitext(r)[0] in name]
    if not matches:
        raise ValueError(f"No measured response matches the run folder {name}")
    return max(matches, key=len)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--responses", required=True, help="folder of measured body responses (.wav)")
    parser.add_argument("runs", nargs="+", help="run folders holding config.yaml and final_state.pth")
    args = parser.parse_args()

    responses = sorted(f for f in os.listdir(args.responses) if f.endswith(".wav"))
    print(f"{'run':40s} {'response':24s} {'MC-LSD':>7s} {'tilt':>7s} {'resonance':>9s}")
    scores = []
    for run in args.runs:
        response = match_response(run, responses)
        learnt, sampling_rate = learnt_response(run)
        freqs, true_db = magnitude_db(measured_response(os.path.join(args.responses, response),
                                                        sampling_rate), sampling_rate)
        scores.append(filter_errors(freqs, true_db, magnitude_db(learnt, sampling_rate)[1]))
        mc_lsd, tilt, resonance = scores[-1]
        print(f"{os.path.basename(os.path.normpath(run)):40s} {response:24s} "
              f"{mc_lsd:7.2f} {tilt:7.2f} {resonance:9.2f}")
    scores = np.array(scores)
    cells = "  ".join(f"{m:.2f} ± {s:.2f}" for m, s in zip(scores.mean(0), scores.std(0)))
    print(f"mean ± std over {len(scores)} runs (MC-LSD, tilt, resonance): {cells}")
