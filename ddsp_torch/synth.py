import math

import torch

from .core import upsample_with_windows
from .dsp import fft_convolve, frequency_to_impulse_response


def harmonic_synth(pitch: torch.Tensor, amplitudes: torch.Tensor, total_amplitude: torch.Tensor,
                   sampling_rate: int, block_size: int) -> torch.Tensor:
    """Additive synthesis of the harmonics of pitch.

    pitch (Hz) and total_amplitude: [batch, frames, 1]; amplitudes: [batch, frames, n_harmonic],
    normalised over the harmonics. Returns [batch, frames * block_size, 1].
    """
    n_samples = pitch.shape[1] * block_size
    amplitudes = upsample_with_windows(amplitudes * total_amplitude, n_samples)
    harmonic_numbers = torch.arange(1, amplitudes.shape[-1] + 1, device=pitch.device, dtype=pitch.dtype)
    frequencies = upsample_with_windows(pitch * harmonic_numbers.view(1, 1, -1), n_samples)
    phase = torch.cumsum(2 * math.pi * frequencies / float(sampling_rate), dim=1)
    return (torch.sin(phase) * amplitudes).sum(dim=-1, keepdim=True)


def filtered_noise_synth(noise_magnitudes: torch.Tensor, block_size: int) -> torch.Tensor:
    """Uniform noise filtered frame by frame with responses built from the magnitudes
    [batch, frames, n_bands]. Returns [batch, frames * block_size, 1]."""
    batch_size, n_frames, _ = noise_magnitudes.shape
    impulse_responses = frequency_to_impulse_response(noise_magnitudes)
    noise = torch.rand(batch_size, n_frames * block_size, device=impulse_responses.device) * 2 - 1
    return fft_convolve(noise, impulse_responses).unsqueeze(-1)
