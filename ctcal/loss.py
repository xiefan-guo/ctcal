"""The CTCal objective (paper Eq. 7).

    L_CTCal = 1/N_noun * sum_{y_i in Y_noun} [
          lambda1 * D(A_stu,y_i, A_tea,y_i)                         # pixel level
        + lambda2 * D(f_enc(A_stu,y_i), f_enc(A_tea,y_i))           # semantic level
        + lambda3 * D(f_dec(f_enc(A_tea,y_i)), A_tea,y_i)           # reconstruction proxy
      ] + lambda4 * R_subject                                       # regularisation

D is MSE throughout. A_tea comes from the smaller timestep and is always detached:
the gradient flows only into the student pass at t_stu.

Two preprocessing details matter here:
  * the regularisation term re-weights attention over semantically relevant tokens
    via a softmax, so responses are comparable across subjects;
  * the alignment terms min-max normalise each map to [0, 1], because response
    magnitude varies with timestep while CTCal only cares about spatial layout.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ctcal import config


def normalize_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Min-max normalise each map in a batch to [0, 1]."""
    flat = x.flatten(1)
    lo = flat.min(dim=1)[0].view(-1, 1, 1, 1)
    hi = flat.max(dim=1)[0].view(-1, 1, 1, 1)
    return (x - lo) / (hi - lo + eps)


def reweight_over_tokens(
    maps: torch.Tensor, valid_len: int | None = None, temperature: float = 100.0
) -> torch.Tensor:
    """Softmax across the token axis so per-subject responses are comparable.

    maps: (H, W, n_tokens) -> same shape. The temperature sharpens the otherwise
    very flat joint-attention distribution. FLUX pads the T5 sequence to a fixed
    length, so `valid_len` restricts the softmax to the real tokens -- including
    the padding columns would flatten the distribution and make the regulariser
    meaningless.
    """
    x = maps.float()
    if valid_len is not None:
        x = x[..., :valid_len]
    return torch.softmax(x * temperature, dim=-1)


class CTCalLoss:
    """Computes L_CTCal for a batch of prompts.

    Args:
        autoencoder: the shared lightweight autoencoder (f_enc / f_dec).
        smoothing: Gaussian blur applied to every map before comparison.
        lambdas: (lambda1..lambda4) trade-off weights; these default to `ctcal.config`.
        tau: threshold in the subject response regularisation (Eq. 6).
    """

    def __init__(
        self,
        autoencoder,
        smoothing,
        lambda1: float = config.LAMBDA1,
        lambda2: float = config.LAMBDA2,
        lambda3: float = config.LAMBDA3,
        lambda4: float = config.LAMBDA4,
        tau: float = config.TAU,
    ):
        self.autoencoder = autoencoder
        self.smoothing = smoothing
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.tau = tau

    def _prepare(
        self, maps: torch.Tensor, token_idxs: list[list[int]], normalize: bool = True
    ) -> torch.Tensor:
        """Pull out one (1, H, W) smoothed map per selected word, stacked.

        `maps` is (H, W, n_tokens) for a single prompt. A word spanning several
        T5 sub-tokens is reduced by averaging over its positions.

        The alignment terms want `normalize=True`, since response magnitude varies
        with timestep. The subject response regulariser wants `normalize=False`:
        per-map min-max scaling would force every peak to exactly 1.0 and erase the
        very imbalance Eq. 6 measures.
        """
        picked = []
        for idxs in token_idxs:
            if not idxs:
                continue
            picked.append(maps[:, :, idxs].mean(dim=-1))  # (H, W)
        if not picked:
            return torch.empty(0, device=maps.device)
        stacked = self.smoothing(torch.stack(picked, dim=0).unsqueeze(1))  # (n, 1, H, W)
        return normalize_map(stacked) if normalize else stacked

    def __call__(
        self,
        attn_stu: torch.Tensor,
        attn_tea: torch.Tensor,
        token_idxs_per_prompt: list[list[list[int]]],
        lambda_t: torch.Tensor,
        valid_lens: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            attn_stu / attn_tea: (B, H, W, n_tokens) aggregated A_IT maps.
            token_idxs_per_prompt: per prompt, per selected word, its token positions.
            lambda_t: (B,) timestep-aware weight, t_stu / T_train (Eq. 8).
            valid_lens: per prompt, the number of non-padding text tokens.
        Returns the scalar loss and a dict of component values for logging.
        """
        device = attn_stu.device
        total = torch.zeros((), device=device, dtype=torch.float32)
        logs = {"pixel": 0.0, "semantic": 0.0, "recon": 0.0, "subject": 0.0, "n_words": 0.0}

        counted = 0
        for i in range(attn_stu.shape[0]):
            token_idxs = token_idxs_per_prompt[i]
            if not any(token_idxs):
                continue

            stu = self._prepare(attn_stu[i], token_idxs)
            tea = self._prepare(attn_tea[i], token_idxs).detach()
            if stu.numel() == 0:
                continue

            w_t = lambda_t[i]
            per_prompt = torch.zeros((), device=device, dtype=torch.float32)

            if self.lambda1 > 0:
                l_pixel = F.mse_loss(stu.float(), tea.float())
                per_prompt = per_prompt + self.lambda1 * w_t * l_pixel
                logs["pixel"] += l_pixel.item()

            if self.lambda2 > 0 or self.lambda3 > 0:
                z_stu = self.autoencoder.encode(stu.float())
                # The teacher branch is detached, so its code and reconstruction
                # train the autoencoder without leaking gradient into the model.
                z_tea = self.autoencoder.encode(tea.float())
                l_sem = F.mse_loss(z_stu, z_tea.detach())
                recon = self.autoencoder.decode(z_tea)
                l_rec = F.mse_loss(recon, tea.float())
                per_prompt = (
                    per_prompt + self.lambda2 * w_t * l_sem + self.lambda3 * l_rec
                )
                logs["semantic"] += l_sem.item()
                logs["recon"] += l_rec.item()

            if self.lambda4 > 0:
                # Eq. 6: pull every subject's peak response up towards the strongest.
                # Softmax over the real tokens first so responses are comparable,
                # then take peaks *without* per-map normalisation.
                vlen = valid_lens[i] if valid_lens is not None else None
                rw = reweight_over_tokens(attn_stu[i], valid_len=vlen)
                reg_idxs = [[j for j in idxs if vlen is None or j < vlen] for idxs in token_idxs]
                reg_maps = self._prepare(rw, reg_idxs, normalize=False)
                if reg_maps.numel() == 0:
                    reg_maps = self._prepare(rw, token_idxs, normalize=False)
                peaks = reg_maps.flatten(1).max(dim=1)[0]  # (n_words,)
                s_attn = peaks.max().detach()
                l_sub = F.relu(s_attn - peaks - self.tau).mean()
                per_prompt = per_prompt + self.lambda4 * w_t * l_sub
                logs["subject"] += l_sub.item()

            total = total + per_prompt
            logs["n_words"] += stu.shape[0]
            counted += 1

        if counted == 0:
            return torch.zeros((), device=device, dtype=torch.float32), logs

        for k in ("pixel", "semantic", "recon", "subject"):
            logs[k] /= counted
        logs["n_words"] /= counted
        return total / counted, logs
