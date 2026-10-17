"""CPU checks for the CTCal loss, autoencoder and token alignment.

No FLUX weights needed, so these run fast and catch shape/semantics bugs before
any GPU time is spent.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctcal.loss import CTCalLoss, normalize_map, reweight_over_tokens  # noqa: E402
from ctcal.modules import AttentionAutoencoder, GaussianSmoothing  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


def main():
    torch.manual_seed(0)
    res = 32

    # --- autoencoder shapes -------------------------------------------------
    ae = AttentionAutoencoder(resolution=res)
    x = torch.rand(5, 1, res, res)
    z, rec = ae(x)
    check("ae latent shape", tuple(z.shape) == (5, 16), str(tuple(z.shape)))
    check("ae recon shape", tuple(rec.shape) == (5, 1, res, res), str(tuple(rec.shape)))
    check("ae recon in [0,1]", bool((rec >= 0).all() and (rec <= 1).all()))
    n_param = sum(p.numel() for p in ae.parameters())
    check("ae is lightweight", n_param < 500_000, f"{n_param / 1e3:.0f}K params")

    ae16 = AttentionAutoencoder(resolution=16)
    z16, rec16 = ae16(torch.rand(2, 1, 16, 16))
    check("ae works at 16x16 (SD2.1 variant)", tuple(rec16.shape) == (2, 1, 16, 16))

    # --- normalisation ------------------------------------------------------
    y = normalize_map(torch.randn(4, 1, res, res) * 5 + 3)
    per = y.flatten(1)
    check(
        "normalize_map -> [0,1] per map",
        bool(torch.allclose(per.min(1)[0], torch.zeros(4), atol=1e-5))
        and bool(torch.allclose(per.max(1)[0], torch.ones(4), atol=1e-5)),
    )

    # --- token reweighting --------------------------------------------------
    maps = torch.rand(2, res, res, 7)
    rw = reweight_over_tokens(maps)
    check(
        "reweight sums to 1 over tokens",
        bool(torch.allclose(rw.sum(-1), torch.ones(2, res, res), atol=1e-4)),
    )

    # --- smoothing ----------------------------------------------------------
    sm = GaussianSmoothing(3, 0.5)
    inp = torch.zeros(1, 1, res, res)
    inp[0, 0, 16, 16] = 1.0
    out = sm(inp)
    check("smoothing preserves shape", tuple(out.shape) == (1, 1, res, res))
    check("smoothing conserves mass", abs(out.sum().item() - 1.0) < 1e-4, f"{out.sum().item():.6f}")
    check("smoothing spreads the spike", out[0, 0, 16, 16].item() < 1.0)

    # --- the loss itself (all four terms enabled) ----------------------------
    loss_fn = CTCalLoss(ae, sm, lambda1=0.01, lambda2=1.0, lambda3=0.01, lambda4=0.1, tau=0.05)
    b, n_tok = 2, 9
    stu = torch.rand(b, res, res, n_tok, requires_grad=True)
    tea = torch.rand(b, res, res, n_tok)
    token_idxs = [[[2], [5, 6]], [[1], [4]]]
    lam = torch.tensor([0.9, 0.2])

    val, logs = loss_fn(stu, tea, token_idxs, lam)
    check("loss is finite scalar", val.dim() == 0 and torch.isfinite(val), f"{val.item():.6f}")
    check("logs report word count", logs["n_words"] == 2.0, str(logs["n_words"]))

    val.backward()
    check("gradient reaches student maps", stu.grad is not None and stu.grad.abs().sum() > 0)

    # Identical maps: the alignment terms must vanish.
    same = torch.rand(b, res, res, n_tok)
    _, logs_same = loss_fn(same.clone().requires_grad_(True), same, token_idxs, lam)
    check("pixel loss ~0 when maps identical", logs_same["pixel"] < 1e-8, f"{logs_same['pixel']:.2e}")
    check(
        "semantic loss ~0 when maps identical",
        logs_same["semantic"] < 1e-8,
        f"{logs_same['semantic']:.2e}",
    )

    # A mismatched pair must score strictly worse than a matched one.
    a = torch.zeros(1, res, res, 3)
    a[0, 8, 8, 0] = 1.0
    a[0, 24, 24, 1] = 1.0
    bmap = torch.zeros(1, res, res, 3)
    bmap[0, 24, 24, 0] = 1.0  # swapped vs. `a`
    bmap[0, 8, 8, 1] = 1.0
    idx = [[[0], [1]]]
    one = torch.tensor([1.0])
    _, log_match = loss_fn(a.clone().requires_grad_(True), a, idx, one)
    _, log_swap = loss_fn(a.clone().requires_grad_(True), bmap, idx, one)
    check(
        "pixel loss larger for mismatched layout",
        log_swap["pixel"] > log_match["pixel"],
        f"{log_swap['pixel']:.4f} > {log_match['pixel']:.4f}",
    )

    # Subject regularisation reads peaks of the token-softmax, so "imbalance" means
    # one subject wins the token competition at its location while the other does
    # not; token 2 is a distractor that outcompetes subject 1.
    vlens = [3]
    bal = torch.zeros(1, res, res, 3)
    bal[0, 8, 8, 0] = 5.0
    bal[0, 24, 24, 1] = 5.0
    imb = bal.clone()
    imb[0, 24, 24, 2] = 20.0  # distractor token dominates subject 1's location
    _, log_bal = loss_fn(bal.clone().requires_grad_(True), bal, idx, one, valid_lens=vlens)
    _, log_imb = loss_fn(imb.clone().requires_grad_(True), imb, idx, one, valid_lens=vlens)
    check(
        "subject reg penalises imbalance",
        log_imb["subject"] > log_bal["subject"] and log_bal["subject"] < 1e-6,
        f"imbalanced={log_imb['subject']:.4f} balanced={log_bal['subject']:.4f}",
    )

    # The regulariser must see raw magnitudes: min-max normalising each map would
    # force every peak to 1.0 and the term would always be 0.
    raw = loss_fn._prepare(reweight_over_tokens(imb[0], valid_len=3), [[0], [1]], normalize=False)
    peaks = raw.flatten(1).max(dim=1)[0]
    check(
        "regulariser peaks differ across subjects",
        (peaks.max() - peaks.min()).item() > 1e-3,
        f"peaks={[round(p, 4) for p in peaks.tolist()]}",
    )

    # Empty selection must be a no-op, not a crash.
    v_empty, _ = loss_fn(stu.detach(), tea, [[], []], lam)
    check("no selected tokens -> zero loss", v_empty.item() == 0.0)

    # lambda_t must scale the loss.
    v_lo, _ = loss_fn(stu.detach(), tea, token_idxs, torch.tensor([0.1, 0.1]))
    v_hi, _ = loss_fn(stu.detach(), tea, token_idxs, torch.tensor([1.0, 1.0]))
    check("larger lambda_t -> larger loss", v_hi.item() > v_lo.item(), f"{v_hi.item():.5f} > {v_lo.item():.5f}")

    print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURES: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
