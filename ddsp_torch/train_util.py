"""Model and loss construction for training."""
from .losses import HarmonicResidualLoss, MultiScaleSTFTLoss
from .model import DDSP

LOUD_DB = -50.0    # frames quieter than this are left out of the supervised loss and the evaluation


def initialize_model(config, device):
    """The model described by config['model'], on device."""
    return DDSP(**config['model']).to(device)


def setup_loss_functions(config, device):
    """Multi-scale spectral loss and, for the DDSP-Violin source with loss.hrl_weight > 0, the
    Harmonic Residual Loss.

    Returns (mssl, hrl or None, whether the HRL is used).
    """
    loss = config['loss']
    mssl = MultiScaleSTFTLoss(**loss['mssl']).to(device)
    hrl_weight = float(loss.get('hrl_weight', 0.0))
    use_hrl = config['model'].get('source') is not None and hrl_weight > 0
    hrl = HarmonicResidualLoss(hrl_weight).to(device) if use_hrl else None
    return mssl, hrl, hrl is not None


def calculate_loss(mssl, hrl, model_outputs, target_signal, pitch, loudness, hrl_active):
    """Spectral loss, plus the HRL when active.

    Returns (total loss, spectral loss, HRL value or None, whether the HRL was added).
    """
    target = target_signal.squeeze(-1) if target_signal.dim() > 2 else target_signal
    spectral_loss = mssl(model_outputs['signal'].squeeze(-1), target)
    if not hrl_active:
        return spectral_loss, spectral_loss, None, False
    hrl_loss = hrl(model_outputs['violin_diagnostics']['residuals'], pitch, loudness)
    return spectral_loss + hrl_loss, spectral_loss, hrl_loss.item(), True
