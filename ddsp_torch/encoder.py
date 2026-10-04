import torch
import torch.nn as nn
import torchaudio

# FFT size and overlap of the MFCC frames for each z_time_steps value, as in DDSP.
STFT_SETTINGS = {63: (2048, 0.5), 125: (1024, 0.5), 250: (1024, 0.75), 500: (512, 0.75), 1000: (256, 0.75)}


class Encoder(nn.Module):
    """Encode audio into a latent sequence z: MFCCs, instance normalisation, a GRU and a
    linear layer, interpolated in time to target_length frames."""

    def __init__(self, rnn_channels: int, z_dims: int, z_time_steps: int, sample_rate: int,
                 target_length: int, n_mfcc: int = 30, n_mels: int = 128, f_min: float = 20.0,
                 f_max: float = 8000.0):
        super().__init__()
        self.target_length = target_length
        fft_size, overlap = STFT_SETTINGS[z_time_steps]
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs={'n_fft': fft_size, 'n_mels': n_mels, 'hop_length': int(fft_size * (1 - overlap)),
                       'f_min': f_min, 'f_max': f_max},
        )
        self.norm = nn.InstanceNorm1d(n_mfcc, eps=1e-5)
        self.gru = nn.GRU(n_mfcc, rnn_channels, batch_first=True)
        self.dense_out = nn.Linear(rnn_channels, z_dims)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: [batch, samples]. Returns z: [batch, target_length, z_dims]."""
        mfcc = self.norm(self.mfcc_transform(audio)).transpose(1, 2)
        z = self.dense_out(self.gru(mfcc)[0])
        z = nn.functional.interpolate(z.transpose(1, 2), size=self.target_length, mode='linear',
                                      align_corners=False)
        return z.transpose(1, 2)
