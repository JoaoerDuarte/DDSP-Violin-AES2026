"""The harmonic-amplitude encoder and the heads that read the source parameters from its latent."""
import torch
import torch.nn as nn


class HarmonicEncoder(nn.Module):
    """GRU over the log-amplitudes of the first n_harmonics harmonics: [B, T, >= n_harmonics] to a
    latent [B, T, z_dims]."""

    def __init__(self, n_harmonics, z_dims=8, rnn_channels=128):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.in_proj = nn.Linear(n_harmonics, rnn_channels)
        self.gru = nn.GRU(rnn_channels, rnn_channels, batch_first=True)
        self.out_proj = nn.Linear(rnn_channels, z_dims)

    def forward(self, harmonic_amps):
        x = torch.log(harmonic_amps[..., :self.n_harmonics].clamp_min(1e-7))
        return self.out_proj(self.gru(self.in_proj(x))[0])


class Head(nn.Module):
    """Three-layer MLP that reads a source parameter from the harmonic-encoder latent."""

    def __init__(self, in_dims, out_dims, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dims, hidden), nn.LeakyReLU(0.1),
                                 nn.Linear(hidden, hidden), nn.LeakyReLU(0.1),
                                 nn.Linear(hidden, out_dims))

    def forward(self, z):
        return self.net(z)
