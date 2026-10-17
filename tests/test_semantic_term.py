"""Check the semantic-level term is well-scaled, not structurally dead.

At initialisation a random autoencoder maps similar inputs to nearly identical
codes, so `semantic` can read as ~0 and look broken. What matters is that the term
becomes informative as the reconstruction proxy task trains the encoder -- the
mechanism the paper relies on, and the reason the proxy task exists.

So this trains the autoencoder on the proxy task alone for a few hundred steps with
synthetic two-blob attention maps, then asserts the code distance separates matched
from mismatched layouts.
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctcal.loss import normalize_map  # noqa: E402
from ctcal.modules import AttentionAutoencoder, GaussianSmoothing  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


def blob(res, cy, cx, sigma=3.0):
    y = torch.arange(res).view(-1, 1).float()
    x = torch.arange(res).view(1, -1).float()
    return torch.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma**2))


def random_map(res, gen):
    cy = torch.randint(4, res - 4, (1,), generator=gen).item()
    cx = torch.randint(4, res - 4, (1,), generator=gen).item()
    return blob(res, cy, cx).unsqueeze(0), (cy, cx)


def main():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    res = 32
    ae = AttentionAutoencoder(resolution=res)
    sm = GaussianSmoothing(3, 0.5)

    def code_dist(a, b):
        za, zb = ae.encode(a), ae.encode(b)
        return F.mse_loss(za, zb).item()

    m1, _ = random_map(res, gen)
    m2, _ = random_map(res, gen)
    a = normalize_map(sm(m1.unsqueeze(0)))
    b = normalize_map(sm(m2.unsqueeze(0)))

    before_same = code_dist(a, a.clone())
    before_diff = code_dist(a, b)
    print(f"  at init: d(same)={before_same:.3e}  d(diff)={before_diff:.3e}")
    check("identical maps give identical codes", before_same < 1e-10)

    # Train the autoencoder on the reconstruction proxy task only.
    opt = torch.optim.AdamW(ae.parameters(), lr=1e-3)
    losses = []
    for step in range(600):
        batch = torch.cat([random_map(res, gen)[0].unsqueeze(0) for _ in range(16)], dim=0)
        batch = normalize_map(sm(batch))
        _, rec = ae(batch)
        loss = F.mse_loss(rec, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"  recon loss: {losses[0]:.4f} -> {sum(losses[-20:]) / 20:.4f}")
    check("proxy task reduces recon loss", sum(losses[-20:]) / 20 < losses[0] * 0.5)

    after_same = code_dist(a, a.clone())
    after_diff = code_dist(a, b)
    print(f"  trained: d(same)={after_same:.3e}  d(diff)={after_diff:.3e}")
    check("trained encoder separates layouts", after_diff > after_same, f"{after_diff:.3e} > {after_same:.3e}")
    check(
        "semantic term grows with training",
        after_diff > before_diff * 10,
        f"{after_diff:.3e} vs {before_diff:.3e} at init",
    )

    # Encoder must not have collapsed to a constant code -- the failure mode the
    # paper's proxy task is designed to prevent.
    codes = torch.cat(
        [ae.encode(normalize_map(sm(random_map(res, gen)[0].unsqueeze(0)))) for _ in range(64)]
    )
    spread = codes.std(dim=0).mean().item()
    print(f"  mean per-dim code std over 64 maps: {spread:.4f}")
    check("encoder has not collapsed", spread > 1e-3, f"std={spread:.4f}")

    # Code distance should track spatial distance between blob centres.
    ref, (ry, rx) = random_map(res, gen)
    ref = normalize_map(sm(ref.unsqueeze(0)))
    pairs = []
    for dy in (1, 4, 10, 20):
        cy = min(max(ry + dy, 0), res - 1)
        other = normalize_map(sm(blob(res, cy, rx).unsqueeze(0).unsqueeze(0)))
        pairs.append((dy, code_dist(ref, other)))
    print("  shift -> code distance:", [(d, round(v, 5)) for d, v in pairs])
    check(
        "code distance grows with spatial offset",
        pairs[-1][1] > pairs[0][1],
        f"{pairs[-1][1]:.5f} > {pairs[0][1]:.5f}",
    )

    print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURES: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
