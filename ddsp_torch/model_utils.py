"""Names of the decoder inputs and outputs, and the DDSP-Violin harmonic source: bow-position
masks, parameter activations and the harmonic distribution."""
import torch

F0_SCALED = 'f0_scaled'
LD_SCALED = 'ld_scaled'
Z = 'z'
AMPS = 'amps'
HARMONIC_DISTRIBUTION = 'harmonic_distribution'
NOISE_MAGNITUDES = 'noise_magnitudes'
BOW_LOGITS = 'bow_logits'
BOW_POSITION_RAW = 'bow_position_raw'
NOTCH_DEPTH_RAW = 'notch_depth_raw'
BRIGHTNESS_RAW = 'brightness_raw'
RESIDUALS_RAW = 'residuals_raw'

NODE_ORDERS = (0, 2, 3, 4, 5)   # bow-position classes: 0 = away from nodes, k = node at x_b = 1/k
CALIBRATED_DECAY = 0.06235      # δ of the calibrated mask, fitted on the dataset
CALIBRATED_POWER = 10           # p of the calibrated mask, sharpens the notch


def scaled_sigmoid(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Map raw decoder outputs to [low, high] with a sigmoid."""
    return low + (high - low) * torch.sigmoid(raw)


def activate_brightness(raw, alpha_min, alpha_max):
    """Brightness exponent α in [alpha_min, alpha_max], decreasing with the raw output."""
    return alpha_max - (alpha_max - alpha_min) * torch.sigmoid(raw)


def activate_residuals(raw, range_db):
    """Residual gain per harmonic, 10^(range_db / 20 * tanh(raw)), within ±range_db dB."""
    return torch.pow(10.0, (range_db / 20.0) * torch.tanh(raw))


def calibrated_terms(beta, n):
    """Depth decay β^(δn) and node selector (1 - |sin(π n β)|)^p of the calibrated mask."""
    decay = beta ** (CALIBRATED_DECAY * n)
    selector = (1.0 - torch.abs(torch.sin(torch.pi * n * beta))) ** CALIBRATED_POWER
    return decay, selector


def node_patterns(mask_type, n_harmonics, device, dtype):
    """Suppression pattern [len(NODE_ORDERS), H] of each class; the class mask is 1 - pattern.

    'calibrated': (1/k)^(δn) (1 - |sin(π n / k)|)^p.  'plain': 1 - |sin(π n / k)|.
    """
    n = torch.arange(1, n_harmonics + 1, device=device, dtype=dtype)
    patterns = torch.zeros(len(NODE_ORDERS), n_harmonics, device=device, dtype=dtype)
    for i, k in enumerate(NODE_ORDERS):
        if k == 0:
            continue
        beta = 1.0 / k
        if mask_type == 'calibrated':
            decay, selector = calibrated_terms(beta, n)
            patterns[i] = decay * selector
        elif mask_type == 'plain':
            patterns[i] = 1.0 - torch.abs(torch.sin(torch.pi * n * beta))
        else:
            raise ValueError(f"bow_mask_type must be 'calibrated' or 'plain', got {mask_type!r}")
    return patterns


def discrete_bow_mask(logits, mask_type, n_harmonics, training):
    """Bow-position mask [B, T, H] from the class logits [B, T, C]: the probability-weighted mixture
    of the class masks in training, the mask of the most likely class otherwise.
    Returns (mask, class probabilities, index of the most likely class)."""
    probs = torch.softmax(logits, dim=-1)
    masks = 1.0 - node_patterns(mask_type, n_harmonics, logits.device, logits.dtype)
    classes = torch.argmax(probs, dim=-1)
    mask = torch.einsum('btc,ch->bth', probs, masks) if training else masks[classes]
    return mask, probs, classes


def continuous_bow_mask(beta, gamma, n):
    """Calibrated mask at a continuous bow position β [B, T, 1] with notch depth γ [B, T, 1]."""
    beta, gamma = beta.expand(-1, -1, n.shape[-1]), gamma.expand(-1, -1, n.shape[-1])
    decay, selector = calibrated_terms(beta, n)
    return 1.0 - gamma * decay * selector


def harmonic_distribution(n, alpha, bow_mask, residuals, pitch, sampling_rate, bow_mask_n_harmonics):
    """Harmonic amplitudes [B, T, H] that sum to one: 1/n * n^α * bow mask * residual gains, with the
    bow mask on the first bow_mask_n_harmonics harmonics only and harmonics above Nyquist removed.

    n is the harmonic number [B, T, H], alpha [B, T, 1], residuals [B, T, <= H], pitch [B, T, 1] in Hz.
    """
    if 0 < bow_mask_n_harmonics < n.shape[-1]:
        bow_mask = bow_mask.clone()
        bow_mask[:, :, bow_mask_n_harmonics:] = 1.0
    spectrum = 1.0 / n * bow_mask * n ** alpha
    gains = torch.ones_like(spectrum)
    gains[:, :, :residuals.shape[-1]] = residuals
    spectrum = spectrum * gains * (pitch * n < sampling_rate / 2.0).to(spectrum.dtype)
    total = spectrum.sum(dim=-1, keepdim=True)
    fallback = torch.zeros_like(spectrum)
    fallback[:, :, 0] = 1.0
    return torch.where(total.expand_as(spectrum) > 1e-7, spectrum / (total + 1e-7), fallback)
