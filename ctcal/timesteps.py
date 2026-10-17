"""Timestep sampling for CTCal on a flow-matching model (paper Sec. 3.3).

For t_stu we keep the model's native training-time sampler. FLUX is a rectified-flow
model trained with a logit-normal sampler, where sigma = sigmoid(X) with
X ~ N(mu, sigma^2), followed by a resolution-dependent shift.

For t_tea the paper warns that naively taking t_tea = 0 is wrong once the sampler is
non-uniform, and defines the priority w_t = p_t / t. On FLUX that formula selects a
sigma above the median t_stu, which would break the method's own t_tea < t_stu
premise: the shift concentrates sampling mass at high noise. So t_tea is instead
taken from `scripts/diagnose_attention.py`, which measures where attention actually
aligns best with the image -- the same evidence the paper's premise rests on.
"""

from __future__ import annotations

import torch


def apply_shift(s, shift: float):
    """FLUX's static sigma shift: s' = shift*s / (1 + (shift-1)*s)."""
    return shift * s / (1.0 + (shift - 1.0) * s)


def sample_student_sigmas(
    batch_size: int,
    mu: float = 0.0,
    sigma: float = 1.0,
    shift: float = 3.0,
    device="cpu",
    generator=None,
) -> torch.Tensor:
    """Draw t_stu from FLUX's native logit-normal + shift schedule."""
    x = torch.randn(batch_size, device=device, generator=generator) * sigma + mu
    return apply_shift(torch.sigmoid(x), shift)


def teacher_sigma_from_diagnostic(
    path: str, metric: str = "vae_corr", max_sigma: float = 0.5
) -> float | None:
    """The sigma the diagnostic measured as most reliable, or None if unmeasured.

    Scripts call this before training; without a diagnostic to read, the caller
    falls back to its own default.
    """
    import json
    import os

    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = json.load(f).get("rows", [])
    usable = [r for r in rows if r["sigma"] <= max_sigma and metric in r]
    if not usable:
        return None
    return float(max(usable, key=lambda r: r[metric])["sigma"])


def lambda_t(sigmas: torch.Tensor) -> torch.Tensor:
    """Timestep-aware adaptive weight (Eq. 8), lambda_t = t_stu / T_train.

    For a flow model sigma already lives in [0, 1] and plays the role of t / T_train
    (sigma = 1 is pure noise), so it is the weight directly.
    """
    return sigmas.clamp(0.0, 1.0)
