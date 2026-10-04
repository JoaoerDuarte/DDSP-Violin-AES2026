import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import lfilter

from .dsp import fft_convolve


def body_filter(resonance_type: str = "convolutional", resonance_length: int = 2048,
                resonance_ar_order: int = 64, resonance_ma_order: int = 64,
                resonance_window_size: int = 1600, resonance_max_reflection: float = 1.0) -> nn.Module:
    """The learnable body filter: FIR ("convolutional", resonance_length taps) or IIR ("ar" or
    "arma", applied to frames of resonance_window_size samples)."""
    if resonance_type == "convolutional":
        return ConvolutionalFilter(resonance_length)
    if resonance_type == "ar":
        return ARFilter(resonance_ar_order, resonance_window_size, resonance_max_reflection)
    if resonance_type == "arma":
        return ARMAFilter(resonance_ar_order, resonance_ma_order, resonance_window_size,
                          resonance_max_reflection)
    raise ValueError(f"resonance_type must be convolutional, ar or arma, not {resonance_type!r}")


class ConvolutionalFilter(nn.Module):
    """Learnable FIR filter, applied by FFT convolution."""

    def __init__(self, length: int):
        super().__init__()
        self.ir = nn.Parameter(torch.randn(length) * 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, samples, 1]. Returns the filtered signal, same shape."""
        ir = self.ir.unsqueeze(0).repeat(x.shape[0], 1)
        return fft_convolve(x.squeeze(-1), ir, delay_compensation=0).unsqueeze(-1)

    def impulse_response(self) -> torch.Tensor:
        return self.ir


def reflection_to_ar(k: torch.Tensor) -> torch.Tensor:
    """Direct-form AR coefficients [1, a_1, ..., a_Q] from reflection coefficients k [Q]
    (step-up recursion). Reflection coefficients inside (-1, 1) give a stable filter."""
    k = k.clone().unsqueeze(0)
    order = k.shape[1]
    a = torch.zeros(1, order + 1, device=k.device, dtype=k.dtype)
    a[:, 0] = 1.0
    for i in range(order):
        a_previous = a[:, :i + 1].clone()
        a[:, i + 1] = k[:, i] * a_previous[:, 0]
        if i > 0:
            indices = torch.arange(1, i + 1, device=k.device)
            a[:, indices] = a_previous[:, indices] + k[:, i:i + 1] * a_previous[:, i - indices + 1]
    return a.squeeze(0)


class IIRFilter(nn.Module):
    """Learnable IIR filter, applied to Hann-windowed frames of window_size samples (hop of a
    quarter frame), each filtered from rest and overlap-added. The output lags the input by
    window_size // 2 samples. The AR part is parametrised by reflection coefficients
    max_reflection * tanh(.), which keeps the filter stable."""

    def __init__(self, window_size: int, max_reflection: float):
        super().__init__()
        self.window_size = window_size
        self.hop_length = window_size // 4
        self.max_reflection = max_reflection
        window = torch.hann_window(window_size)
        self.register_buffer('_kernel', torch.diag(window).unsqueeze(1), persistent=False)

    def coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Denominator and numerator coefficients (a, b) of equal length, a[0] = 1."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, samples, 1]. Returns the filtered signal, same shape."""
        a, b = self.coefficients()
        x = x.squeeze(-1)
        signal_length = x.shape[1]
        padding = self.window_size // 2
        frames = F.pad(x, (padding, padding), 'constant', 0).unfold(-1, self.window_size, self.hop_length)
        batch, n_frames, window = frames.shape
        filtered = lfilter(frames.reshape(-1, window), a.expand(batch * n_frames, -1),
                           b.expand(batch * n_frames, -1))
        filtered = filtered.reshape(batch, n_frames, window).transpose(1, 2)
        ones = torch.ones(1, filtered.shape[1], filtered.shape[2], device=filtered.device)
        result = F.conv_transpose1d(torch.cat([filtered, ones], dim=0), self._kernel,
                                    stride=self.hop_length, padding=0).squeeze(1)
        output, normalization = result[:-1], result[-1:]
        output = output / (normalization + 1e-8)
        return output[:, :signal_length].unsqueeze(-1)

    def impulse_response(self) -> torch.Tensor:
        """Response of the filter, from rest, to a unit impulse (window_size samples)."""
        a, b = self.coefficients()
        impulse = torch.zeros(self.window_size, device=a.device)
        impulse[0] = 1.0
        return lfilter(impulse, a, b)


class ARFilter(IIRFilter):
    """All-pole filter gain / A(z) of the given order."""

    def __init__(self, order: int, window_size: int, max_reflection: float):
        super().__init__(window_size, max_reflection)
        self.reflection_coeffs = nn.Parameter(torch.zeros(order))
        self.gain = nn.Parameter(torch.ones(1) * 0.1)

    def coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.max_reflection * torch.tanh(self.reflection_coeffs)
        ar_coeffs = reflection_to_ar(k)[1:]
        a = torch.cat([torch.ones(1, device=ar_coeffs.device), ar_coeffs])
        b = torch.zeros_like(a)
        b[0] = self.gain
        return a, b


class ARMAFilter(IIRFilter):
    """Pole-zero filter B(z) / A(z) with ar_order poles and ma_order zeros. B starts with the gain."""

    def __init__(self, ar_order: int, ma_order: int, window_size: int, max_reflection: float):
        super().__init__(window_size, max_reflection)
        self.ar_reflection_coeffs = nn.Parameter(torch.zeros(ar_order))
        self.ma_coeffs = nn.Parameter(torch.randn(ma_order) * 1e-4)
        self.gain = nn.Parameter(torch.ones(1) * 0.1)

    def coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.max_reflection * torch.tanh(self.ar_reflection_coeffs)
        ar_coeffs = reflection_to_ar(k)[1:]
        a = torch.cat([torch.ones(1, device=ar_coeffs.device), ar_coeffs])
        b = torch.cat([self.gain, self.ma_coeffs])
        if len(a) < len(b):
            a = F.pad(a, (0, len(b) - len(a)))
        elif len(b) < len(a):
            b = F.pad(b, (0, len(a) - len(b)))
        return a, b
