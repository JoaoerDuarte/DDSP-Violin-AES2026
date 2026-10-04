import torch
import torch.nn as nn

from .core import multiscale_fft, safe_log, scale_db


class MultiScaleSTFTLoss(nn.Module):
    """Sum over FFT sizes of the mean absolute differences of linear and log STFT magnitudes."""

    def __init__(self, scales: list[int], overlap: float):
        super().__init__()
        self.scales = scales
        self.overlap = overlap

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_mags = multiscale_fft(target, self.scales, self.overlap)
        predicted_mags = multiscale_fft(predicted, self.scales, self.overlap)
        total_loss = 0.0
        for target_mag, predicted_mag in zip(target_mags, predicted_mags):
            linear_loss = (target_mag - predicted_mag).abs().mean()
            log_loss = (safe_log(target_mag) - safe_log(predicted_mag)).abs().mean()
            total_loss = total_loss + linear_loss + log_loss
        return total_loss


class HarmonicResidualLoss(nn.Module):
    """Harmonic Residual Loss: weight times the mean squared deviation of the residual
    corrections from 1, over the frames whose scaled loudness exceeds loudness_threshold
    and whose f0 exceeds pitch_threshold (Hz)."""

    def __init__(self, weight: float, loudness_threshold: float = 0.2, pitch_threshold: float = 20.0):
        super().__init__()
        self.weight = weight
        self.loudness_threshold = loudness_threshold
        self.pitch_threshold = pitch_threshold

    def forward(self, residuals: torch.Tensor, f0_hz: torch.Tensor, loudness: torch.Tensor) -> torch.Tensor:
        """residuals: [batch, frames, n_residuals]; f0_hz and loudness (dB): [batch, frames, 1]."""
        valid = torch.logical_and(scale_db(loudness.squeeze(-1)) > self.loudness_threshold,
                                  f0_hz.squeeze(-1) > self.pitch_threshold)
        if not torch.any(valid):
            return residuals.new_zeros(())
        deviation = torch.abs(residuals - 1.0) ** 2
        return self.weight * deviation[valid.unsqueeze(-1).expand_as(residuals)].mean()
