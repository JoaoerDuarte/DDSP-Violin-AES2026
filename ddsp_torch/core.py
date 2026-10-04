import math

import librosa
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F

DB_RANGE = 80.0
F0_RANGE = 127.0
F0_SEARCH_HZ = (180.0, 1400.0)  # pitch range searched when a file gives no nominal pitch


def overlap_and_add(signal: torch.Tensor, frame_step: int) -> torch.Tensor:
    """Overlap-add frames [..., n_frames, frame_length] placed frame_step samples apart."""
    outer_dimensions = signal.shape[:-2]
    n_frames = signal.shape[-2]
    frame_length = signal.shape[-1]
    output_length = frame_length + frame_step * (n_frames - 1)

    segments = math.ceil(frame_length / frame_step)
    pad_length = (segments * frame_step) - frame_length
    signal = F.pad(signal, (0, pad_length, 0, segments))

    shape = outer_dimensions + (n_frames + segments, segments, frame_step)
    signal = signal.reshape(shape)
    dims = list(range(signal.dim()))
    dims = dims[:-3] + [dims[-2], dims[-3], dims[-1]]
    signal = signal.permute(*dims)

    shape = outer_dimensions + ((n_frames + segments) * segments, frame_step)
    signal = signal.reshape(shape)
    signal = signal[..., :(n_frames + segments - 1) * segments, :]

    shape = outer_dimensions + (segments, n_frames + segments - 1, frame_step)
    signal = signal.reshape(shape)
    signal = torch.sum(signal, dim=-3)
    return signal.reshape(*outer_dimensions, -1)[..., :output_length]


def scale_db(db: torch.Tensor) -> torch.Tensor:
    """Map loudness in dB from [-DB_RANGE, 0] to [0, 1]."""
    return torch.clamp((db / DB_RANGE) + 1.0, 0.0, 1.0)


def hz_to_midi(f0_hz: torch.Tensor) -> torch.Tensor:
    """Convert Hz to MIDI note numbers (A4 = 440 Hz = 69). 0 Hz maps to 0."""
    valid = f0_hz > 0
    log_term = torch.log(torch.where(valid, f0_hz, 1e-7) / 440.0) / torch.log(torch.tensor(2.0))
    return torch.where(valid, 12.0 * log_term + 69.0, 0.0)


def scale_f0_hz(f0_hz: torch.Tensor) -> torch.Tensor:
    """Map f0 in Hz to [0, 1] through MIDI note numbers 0 to F0_RANGE."""
    return torch.clamp(hz_to_midi(f0_hz), 0.0, F0_RANGE) / F0_RANGE


def safe_log(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return torch.log(x + eps)


def multiscale_fft(signal: torch.Tensor, scales: list[int], overlap: float) -> list[torch.Tensor]:
    """STFT magnitudes of signal [batch, samples] for each FFT size in scales."""
    stfts = []
    for scale in scales:
        stft = torch.stft(
            signal,
            n_fft=scale,
            hop_length=int(scale * (1 - overlap)),
            win_length=scale,
            window=torch.hann_window(scale, device=signal.device),
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        stfts.append(torch.abs(stft))
    return stfts


def upsample_with_windows(inputs: torch.Tensor, n_timesteps: int) -> torch.Tensor:
    """Upsample frames [batch, n_frames, channels] to n_timesteps samples by
    overlap-adding Hann windows centred on the frames, as in DDSP."""
    batch_size, n_frames, n_channels = inputs.shape
    inputs = torch.cat([inputs, inputs[:, -1:, :]], dim=1)
    hop_size = n_timesteps // n_frames
    window = torch.hann_window(2 * hop_size, device=inputs.device)
    windowed = inputs.permute(0, 2, 1).reshape(-1, n_frames + 1, 1) * window.view(1, 1, -1)
    output = overlap_and_add(windowed, hop_size)
    output = output.reshape(batch_size, n_channels, -1).permute(0, 2, 1)
    return output[:, hop_size:-hop_size, :]


def remove_above_nyquist(amplitudes: torch.Tensor, pitch: torch.Tensor, sampling_rate: float) -> torch.Tensor:
    """Zero the amplitudes of harmonics at or above the Nyquist frequency."""
    n_harmonics = amplitudes.shape[-1]
    harmonic_numbers = torch.arange(1, n_harmonics + 1, device=amplitudes.device, dtype=amplitudes.dtype)
    harmonic_freqs = pitch * harmonic_numbers
    mask = (harmonic_freqs < sampling_rate / 2.0).to(amplitudes.dtype)
    return amplitudes * mask


def exp_sigmoid(x: torch.Tensor, exponent: float = 10.0, max_value: float = 2.0,
                threshold: float = 1e-7) -> torch.Tensor:
    """max_value * sigmoid(x) ** log(exponent) + threshold."""
    exponent = torch.tensor(exponent, device=x.device, dtype=torch.float32)
    max_value = torch.tensor(max_value, device=x.device, dtype=torch.float32)
    threshold = torch.tensor(threshold, device=x.device, dtype=torch.float32)
    return max_value * torch.sigmoid(x)**torch.log(exponent) + threshold


def extract_loudness(signal: np.ndarray, sampling_rate: int, block_size: int, n_fft: int = 2048,
                     range_db: float = DB_RANGE, ref_db: float = 20.0) -> np.ndarray:
    """A-weighted loudness in dB per frame of block_size samples, floored at -range_db."""
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=block_size, win_length=n_fft, center=True,
                        pad_mode='reflect')
    power_spectrum = np.abs(stft)**2
    with np.errstate(divide='ignore'):  # log of the 0 Hz bin, clipped by A_weighting
        a_weighting = librosa.A_weighting(librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft))
    weighted_power_db = 10.0 * np.log10(power_spectrum + 1e-10) + a_weighting[:, np.newaxis]
    loudness_db = np.mean(weighted_power_db, axis=0)
    loudness_db -= ref_db
    loudness_db = np.maximum(loudness_db, -range_db)
    return loudness_db.astype(np.float32)


def extract_pitch(signal: np.ndarray, sampling_rate: int, block_size: int,
                  target_f0: float | None = None) -> np.ndarray:
    """f0 in Hz per frame of block_size samples (0 where unvoiced), from pYIN.

    With the nominal pitch target_f0, the search is limited to 7 semitones (a factor 1.5) around it,
    which removes the octave errors of an unbounded search on bowed-string spectra. Without it, the
    search covers F0_SEARCH_HZ.
    """
    f_min, f_max = F0_SEARCH_HZ
    if target_f0 is not None:
        f_min, f_max = max(f_min, target_f0 / 1.5), min(f_max, target_f0 * 1.5)
    f0, _, _ = librosa.pyin(signal, fmin=f_min, fmax=f_max, sr=sampling_rate, frame_length=2048,
                            hop_length=block_size)
    return np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)


def extract_harmonic_amplitudes(signal: np.ndarray, f0: np.ndarray, sampling_rate: int, block_size: int,
                                n_harmonics: int = 40, n_fft: int = 2048) -> np.ndarray:
    """STFT magnitude at the nearest bin to each harmonic n * f0, per frame: [frames, n_harmonics].

    f0 has one value per frame (0 = unvoiced). Unvoiced frames and harmonics above Nyquist are zero.
    """
    signal = np.asarray(signal, dtype=np.float64)
    f0 = np.asarray(f0, dtype=np.float64)
    window = scipy.signal.get_window('hann', n_fft)
    padded = np.pad(signal, n_fft // 2, mode='reflect')
    bin_width = sampling_rate / n_fft
    amps = np.zeros((len(f0), n_harmonics), dtype=np.float32)
    for t in range(min(len(f0), (len(padded) - n_fft) // block_size + 1)):
        if f0[t] <= 0.0:
            continue
        spectrum = np.abs(np.fft.rfft(padded[t * block_size:t * block_size + n_fft] * window, n=n_fft))
        max_k = min(n_harmonics, int(np.floor((sampling_rate / 2.0 - bin_width / 2.0) / f0[t])))
        if max_k < 1:
            continue
        bins = np.clip(np.round(np.arange(1, max_k + 1) * f0[t] / bin_width).astype(int), 0, len(spectrum) - 1)
        amps[t, :max_k] = spectrum[bins]
    return amps
