import math

import torch
import torch.nn.functional as F

from .core import overlap_and_add


def window_impulse_response(impulse_response: torch.Tensor) -> torch.Tensor:
    """Apply a Hann window to zero-phase impulse responses from irfft and return them in causal form."""
    window = torch.hann_window(impulse_response.shape[-1], dtype=impulse_response.dtype,
                               device=impulse_response.device)
    window = torch.fft.fftshift(window, dim=-1)
    return torch.fft.fftshift(window * impulse_response, dim=-1)


def frequency_to_impulse_response(magnitudes: torch.Tensor) -> torch.Tensor:
    """Windowed impulse responses [..., 2 * (n_bands - 1)] from magnitude responses [..., n_bands]."""
    spectrum = torch.view_as_complex(torch.stack([magnitudes, torch.zeros_like(magnitudes)], -1))
    return window_impulse_response(torch.fft.irfft(spectrum))


def fft_convolve(audio: torch.Tensor, impulse_response: torch.Tensor,
                 delay_compensation: int = -1) -> torch.Tensor:
    """Convolve audio [batch, samples] with impulse responses by FFT and overlap-add.

    impulse_response is [batch, frames, taps] or [frames, taps] (the same for the whole
    batch): the audio is cut into as many frames, each convolved with its own response.
    The output keeps the input length and starts delay_compensation samples into the
    full convolution ((taps - 1) // 2 when negative).
    """
    batch_size, audio_size = audio.shape
    if impulse_response.dim() == 2:
        impulse_response = impulse_response.unsqueeze(0)
    if impulse_response.shape[0] == 1 and batch_size > 1:
        impulse_response = impulse_response.expand(batch_size, -1, -1)
    n_frames, ir_size = impulse_response.shape[-2:]

    frame_size = math.ceil(audio_size / n_frames)
    audio = F.pad(audio, (0, n_frames * frame_size - audio_size))
    audio_frames = audio.unfold(-1, frame_size, frame_size)
    fft_size = 2**math.ceil(math.log2(ir_size + frame_size - 1))
    audio_fft = torch.fft.rfft(audio_frames, n=fft_size)
    ir_fft = torch.fft.rfft(impulse_response, n=fft_size)
    output = overlap_and_add(torch.fft.irfft(audio_fft * ir_fft, n=fft_size), frame_size)

    start = (ir_size - 1) // 2 if delay_compensation < 0 else delay_compensation
    return output[..., start:start + audio_size]
