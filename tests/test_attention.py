"""Check the CTCal attention processor against stock FLUX attention.

Two things must hold: installing the processor does not change the transformer's
output, and the stored A_IT maps have the expected shape and are proper
distributions over the joint key axis restricted to text columns.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctcal import config  # noqa: E402
from ctcal.attention import install_attention_store  # noqa: E402


def main():
    from diffusers import FluxTransformer2DModel

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    print(f"loading transformer from {config.flux_path()} ...", flush=True)
    tr = FluxTransformer2DModel.from_pretrained(
        config.flux_path(), subfolder="transformer", dtype=dtype
    ).to(device)
    tr.eval()

    # Synthetic batch: 16x16 patches after 2x2 packing.
    b, h, w = 1, 16, 16
    img_len = h * w
    txt_len = 24
    cfg = tr.config
    hidden = torch.randn(b, img_len, cfg.in_channels, device=device, dtype=dtype)
    enc = torch.randn(b, txt_len, cfg.joint_attention_dim, device=device, dtype=dtype)
    pooled = torch.randn(b, cfg.pooled_projection_dim, device=device, dtype=dtype)
    timestep = torch.tensor([500.0], device=device, dtype=dtype)

    txt_ids = torch.zeros(txt_len, 3, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    img_ids = torch.stack(
        [torch.zeros_like(yy).flatten(), yy.flatten(), xx.flatten()], dim=-1
    ).to(dtype)

    kwargs = dict(
        hidden_states=hidden,
        encoder_hidden_states=enc,
        pooled_projections=pooled,
        timestep=timestep,
        # FLUX.1-dev is guidance-distilled and takes a guidance embedding.
        guidance=torch.full((b,), config.DEFAULT_GUIDANCE, device=device, dtype=dtype),
        img_ids=img_ids,
        txt_ids=txt_ids,
        return_dict=False,
    )

    with torch.no_grad():
        ref = tr(**kwargs)[0].float()

    layers = [6, 12, 18]
    store = install_attention_store(tr, layers)

    # Disabled -> must match stock output.
    store.enabled = False
    with torch.no_grad():
        out_off = tr(**kwargs)[0].float()
    d_off = (ref - out_off).abs().max().item()
    print(f"max|stock - ctcal(disabled)| = {d_off:.3e}")

    # Enabled -> output unchanged, maps populated.
    store.enabled = True
    store.detach = True
    store.reset()
    with torch.no_grad():
        out_on = tr(**kwargs)[0].float()
    d_on = (ref - out_on).abs().max().item()
    print(f"max|stock - ctcal(enabled) | = {d_on:.3e}")
    print(f"stored maps: {len(store.maps)} (expected {len(layers)})")
    print(f"per-map shape: {tuple(store.maps[0].shape)}")

    agg = store.aggregate(h, w)
    print(f"aggregated:   {tuple(agg.shape)} (expected ({b}, {h}, {w}, {txt_len}))")

    # Each row of the joint softmax sums to 1, so the text slice sums to <= 1.
    row_txt_mass = store.maps[0].sum(dim=-1)
    print(
        f"text-column mass per image row: min={row_txt_mass.min().item():.4f} "
        f"max={row_txt_mass.max().item():.4f} (must be in (0, 1])"
    )

    ok = (
        d_off < 1e-6
        and d_on < 1e-6
        and len(store.maps) == len(layers)
        and tuple(agg.shape) == (b, h, w, txt_len)
        and row_txt_mass.max().item() <= 1.0 + 1e-3
    )
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
