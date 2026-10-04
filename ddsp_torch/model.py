import torch
import torch.nn as nn

from .core import exp_sigmoid, remove_above_nyquist, scale_db, scale_f0_hz
from .decoder import Decoder
from .encoder import Encoder
from .filters import body_filter
from .harmonic_encoder import HarmonicEncoder, Head
from .model_utils import (
    AMPS, BOW_LOGITS, BOW_POSITION_RAW, BRIGHTNESS_RAW, F0_SCALED, HARMONIC_DISTRIBUTION, LD_SCALED,
    NODE_ORDERS, NOISE_MAGNITUDES, NOTCH_DEPTH_RAW, RESIDUALS_RAW, Z, activate_brightness,
    activate_residuals, continuous_bow_mask, discrete_bow_mask, harmonic_distribution, scaled_sigmoid,
)
from .synth import filtered_noise_synth, harmonic_synth

BETA_RANGE = (1 / 6, 0.5)    # continuous bow position β, between the k = 6 and k = 2 nodes
GAMMA_RANGE = (0.75, 1.0)    # continuous notch depth γ
LATENT_DIMS = 8              # size of the harmonic-encoder latent


class DDSP(nn.Module):
    """DDSP with an MFCC encoder, filtered noise and a learnable body filter (FIR by default,
    AR or ARMA with resonance_type), applied before or after the noise (resonance_position).

    With a source config, the harmonic distribution is the DDSP-Violin source: the Helmholtz 1/n
    decay shaped by a brightness mask n^α, a bow-position mask and per-harmonic residual gains ρ_n.
    With the harmonic-amplitude encoder, α and the bow position come from heads on its latent,
    otherwise from the decoder. The bow position is either a distribution over the node orders
    NODE_ORDERS (discrete head) or a continuous β with notch depth γ. Without a source config, the
    decoder predicts the harmonic distribution freely (the DDSP baseline).
    """

    def __init__(self, sampling_rate: int, block_size: int, signal_length: int, n_harmonic: int,
                 n_bands: int, encoder: dict, decoder: dict, source: dict | None = None,
                 resonance_type: str = "convolutional", resonance_length: int = 2048,
                 resonance_ar_order: int = 64, resonance_ma_order: int = 64,
                 resonance_window_size: int = 1600, resonance_max_reflection: float = 1.0,
                 resonance_position: str = "before_noise"):
        super().__init__()
        if resonance_position not in ("before_noise", "after_noise"):
            raise ValueError(f"resonance_position must be before_noise or after_noise, "
                             f"not {resonance_position!r}")
        self.n_harmonic = n_harmonic
        self.resonance_position = resonance_position
        self.violin_source = source is not None
        use_harmonic_encoder = self.violin_source and bool(source['use_harmonic_encoder'])
        if self.violin_source:
            self.alpha_min, self.alpha_max = (float(a) for a in source['alpha_range'])
            self.residual_range_db = float(source['residual_range_db'])
            self.discrete_head = bool(source['discrete_head'])
            self.bow_mask_type = source['bow_mask_type']
            self.bow_mask_n_harmonics = int(source['bow_mask_n_harmonics'])
            if not self.discrete_head and not (use_harmonic_encoder and self.bow_mask_type == 'calibrated'):
                raise ValueError('the continuous head needs the harmonic encoder and the calibrated mask')
        self.register_buffer("sampling_rate", torch.tensor(float(sampling_rate)))
        self.register_buffer("block_size", torch.tensor(int(block_size)))
        self.register_buffer("noise_bias", torch.tensor(-5.0))

        self.encoder = Encoder(sample_rate=sampling_rate, target_length=signal_length // block_size,
                               **encoder)
        self.harmonic_encoder = (HarmonicEncoder(int(source['harmonic_encoder_n_harmonics']), LATENT_DIMS)
                                 if use_harmonic_encoder else None)
        outputs = [(AMPS, 1)]
        if not self.violin_source:
            outputs.append((HARMONIC_DISTRIBUTION, n_harmonic))
        else:
            if not use_harmonic_encoder:    # the decoder also gives the class logits and the brightness
                outputs += [(BOW_LOGITS, len(NODE_ORDERS)), (BRIGHTNESS_RAW, 1)]
            outputs.append((RESIDUALS_RAW, int(source['n_residuals'])))
        outputs.append((NOISE_MAGNITUDES, n_bands))
        self.decoder = Decoder([F0_SCALED, LD_SCALED, Z], outputs, encoder["z_dims"], **decoder)
        self.resonance = body_filter(resonance_type, resonance_length, resonance_ar_order,
                                     resonance_ma_order, resonance_window_size, resonance_max_reflection)
        # Heads on the harmonic-encoder latent: bow position (class logits, or β and γ) and brightness.
        discrete = self.violin_source and self.discrete_head
        self.beta_gamma_head = (Head(LATENT_DIMS, len(NODE_ORDERS))
                                if use_harmonic_encoder and discrete else None)
        self.continuous_bow_gamma_head = (Head(LATENT_DIMS, 2)
                                          if use_harmonic_encoder and not discrete else None)
        self.brightness_head = Head(LATENT_DIMS, 1) if use_harmonic_encoder else None

    def forward(self, pitch: torch.Tensor, loudness: torch.Tensor, audio: torch.Tensor,
                harmonic_amps: torch.Tensor | None = None) -> dict:
        """pitch (Hz) and loudness (dB) [batch, frames, 1], audio [batch, samples], harmonic
        amplitudes [batch, frames, >= 10] (harmonic-amplitude encoder only).

        Returns the output 'signal' [batch, samples, 1] and, under 'violin_diagnostics', the source
        controls: α, the residual gains, and the class logits, probabilities and most likely class
        (discrete head) or β (continuous head), or the harmonic amplitudes of the baseline.
        """
        if pitch.dim() == 2:
            pitch = pitch.unsqueeze(-1)
        if loudness.dim() == 2:
            loudness = loudness.unsqueeze(-1)
        if audio.dim() == 3:
            audio = audio.squeeze(-1)
        sampling_rate = int(self.sampling_rate.item())
        block_size = int(self.block_size.item())
        inputs = {F0_SCALED: scale_f0_hz(pitch), LD_SCALED: scale_db(loudness), Z: self.encoder(audio)}
        latent = self.harmonic_encoder(harmonic_amps) if self.harmonic_encoder is not None else None
        params = self.decoder(**inputs)
        if latent is not None:
            if self.discrete_head:
                params[BOW_LOGITS] = self.beta_gamma_head(latent)
            else:
                bow = self.continuous_bow_gamma_head(latent)
                params[BOW_POSITION_RAW], params[NOTCH_DEPTH_RAW] = bow[..., 0:1], bow[..., 1:2]
            params[BRIGHTNESS_RAW] = self.brightness_head(latent)
        total_amplitude = exp_sigmoid(params[AMPS])
        noise_magnitudes = exp_sigmoid(params[NOISE_MAGNITUDES] + self.noise_bias, threshold=0.0)

        if self.violin_source:
            amplitudes, diagnostics = self._violin_source(params, pitch, float(sampling_rate))
        else:
            amplitudes = exp_sigmoid(params[HARMONIC_DISTRIBUTION])
            amplitudes = remove_above_nyquist(amplitudes, pitch, float(sampling_rate))
            amplitudes = amplitudes / (amplitudes.sum(-1, keepdim=True) + 1e-7)
            diagnostics = {'harmonic_amplitudes': amplitudes}

        signal = harmonic_synth(pitch, amplitudes, total_amplitude, sampling_rate, block_size)
        if self.resonance_position == "before_noise":
            signal = self.resonance(signal)
        signal = signal + filtered_noise_synth(noise_magnitudes, block_size)
        if self.resonance_position == "after_noise":
            signal = self.resonance(signal)
        return {'signal': signal, 'violin_diagnostics': diagnostics}

    def _violin_source(self, params: dict, pitch: torch.Tensor, sampling_rate: float):
        """Harmonic amplitudes of the DDSP-Violin source and its controls."""
        n = torch.arange(1, self.n_harmonic + 1, device=pitch.device, dtype=pitch.dtype).view(1, 1, -1)
        n = n.expand(pitch.shape[0], pitch.shape[1], -1)
        if self.discrete_head:
            bow_mask, probs, classes = discrete_bow_mask(params[BOW_LOGITS], self.bow_mask_type,
                                                         self.n_harmonic, self.training)
            diagnostics = {'softmax_k_logits': params[BOW_LOGITS], 'softmax_k_probs': probs,
                           'softmax_k_class': classes}
        else:
            beta = scaled_sigmoid(params[BOW_POSITION_RAW], *BETA_RANGE)
            bow_mask = continuous_bow_mask(beta, scaled_sigmoid(params[NOTCH_DEPTH_RAW], *GAMMA_RANGE), n)
            diagnostics = {'β': beta}
        alpha = activate_brightness(params[BRIGHTNESS_RAW], self.alpha_min, self.alpha_max)
        residuals = activate_residuals(params[RESIDUALS_RAW], self.residual_range_db)
        amplitudes = harmonic_distribution(n, alpha, bow_mask, residuals, pitch, sampling_rate,
                                           self.bow_mask_n_harmonics)
        diagnostics.update({'α': alpha, 'residuals': residuals})
        return amplitudes, diagnostics
