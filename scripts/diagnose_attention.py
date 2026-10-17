"""Test the paper's founding observation on FLUX (paper Fig. 1b).

Claim: given a real image / prompt / noise triplet, the cross-attention maps a
diffusion model produces at *small* timesteps align well with the ground-truth image
layout, and that alignment degrades as the timestep grows. CTCal only makes sense if
this holds, and it decides which t_tea to calibrate from.

Alignment is measured without any external detector, using the image itself as the
reference. For a prompt with two nouns we have a reference segmentation problem we
cannot solve exactly, so three complementary proxies are used at each sigma:

  * `peak_sep`    distance between the two nouns' attention peaks. Distinct objects
                  should occupy distinct locations.
  * `iou_overlap` IoU of the two nouns' top-10% response masks. Lower is better.
  * `vae_corr`    correlation between each noun's attention map and the spatial
                  energy of the clean VAE latent.

Also reports `consistency`: the similarity between the map at each sigma and the map
at the smallest sigma, which is the quantity CTCal directly optimises.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctcal import config  # noqa: E402
from ctcal.attention import install_attention_store  # noqa: E402
from ctcal.loss import normalize_map  # noqa: E402
from ctcal.modules import GaussianSmoothing  # noqa: E402
from ctcal.pos_select import get_nlp, select_words, word_to_token_indices  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-roots", nargs="+", required=True, help="reward-selected pairs to probe")
    p.add_argument("--lora", default=None, help="evaluate a fine-tuned checkpoint")
    p.add_argument("--tag", default="baseline")
    p.add_argument("--num-samples", type=int, default=40)
    p.add_argument("--resolution", type=int, default=config.TRAIN_RESOLUTION)
    p.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.98],
    )
    p.add_argument("--attn-layers", type=int, nargs="+", default=config.DEFAULT_ATTN_LAYERS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=os.path.join(config.REPO_ROOT, "results", "diagnose"))
    p.add_argument("--save-maps", type=int, default=3, help="dump this many map grids as PNG")
    return p.parse_args()


def peak_xy(m: np.ndarray):
    return divmod(int(np.argmax(m)), m.shape[1])


def mask_iou(a: np.ndarray, b: np.ndarray, q: float = 0.9):
    ma = a >= np.quantile(a, q)
    mb = b >= np.quantile(b, q)
    union = np.logical_or(ma, mb).sum()
    return float(np.logical_and(ma, mb).sum() / union) if union else 0.0


def _save_grid(maps_by_sigma, sigmas, words, prompt, args, index):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_words = len(maps_by_sigma[sigmas[0]])
    fig, axes = plt.subplots(n_words, len(sigmas), figsize=(2 * len(sigmas), 2 * n_words))
    axes = np.atleast_2d(axes)
    for r in range(n_words):
        for c, sig in enumerate(sigmas):
            ax = axes[r, c]
            ax.imshow(maps_by_sigma[sig][r], cmap="YlOrRd")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"sigma={sig}", fontsize=9)
            if c == 0:
                ax.set_ylabel(words[r] if r < len(words) else "", fontsize=9)
    fig.suptitle(prompt, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"{args.tag}_maps_{index}.png"),
                dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    from diffusers import FluxPipeline

    from ctcal.data import RewardSelectedDataset

    torch.manual_seed(args.seed)
    device = "cuda"
    dtype = torch.bfloat16
    os.makedirs(args.out_dir, exist_ok=True)

    pipe = FluxPipeline.from_pretrained(config.flux_path(), dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    if args.lora:
        pipe.load_lora_weights(args.lora)
        print(f"loaded LoRA from {args.lora}", flush=True)

    transformer, vae = pipe.transformer, pipe.vae
    transformer.eval()
    store = install_attention_store(transformer, args.attn_layers)
    store.detach = True
    smoothing = GaussianSmoothing(config.SMOOTH_KERNEL, config.SMOOTH_SIGMA).to(device)
    nlp = get_nlp(config.stanza_dir())

    dataset = RewardSelectedDataset(args.data_roots, resolution=args.resolution)
    n = min(args.num_samples, len(dataset))
    print(f"probing {n} samples at sigmas {args.sigmas}", flush=True)

    acc = {s: {"peak_sep": [], "iou": [], "vae_corr": [], "consistency": []} for s in args.sigmas}
    saved = 0

    for si in range(n):
        rec = dataset[si]
        prompt = rec["prompt"]
        words = select_words(prompt, nlp)
        token_idxs = word_to_token_indices(prompt, words, pipe.tokenizer_2, config.T5_MAX_LENGTH)
        token_idxs = [t for t in token_idxs if t]
        if len(token_idxs) < 2:
            continue

        pixel_values = rec["pixel_values"].unsqueeze(0).to(device, dtype=dtype)
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
            prompt_embeds, pooled, text_ids = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=config.T5_MAX_LENGTH,
            )

        lh, lw = latents.shape[2], latents.shape[3]
        h, w = lh // 2, lw // 2
        packed = pipe._pack_latents(latents, 1, latents.shape[1], lh, lw)
        img_ids = pipe._prepare_latent_image_ids(1, h, w, device, dtype)
        guidance = torch.full((1,), config.DEFAULT_GUIDANCE, device=device, dtype=dtype)

        # Spatial energy of the clean latent: where the image has structure.
        energy = latents.float().pow(2).mean(dim=1, keepdim=True)
        energy = torch.nn.functional.adaptive_avg_pool2d(energy, (h, w))[0, 0].cpu().numpy()
        energy = (energy - energy.mean()) / (energy.std() + 1e-8)

        # One fixed noise draw across sigmas, so differences are the timestep's doing.
        noise = torch.randn(
            packed.shape, device=device, dtype=dtype,
            generator=torch.Generator(device).manual_seed(args.seed + si),
        )

        maps_by_sigma = {}
        for sig in args.sigmas:
            s = torch.tensor([sig], device=device, dtype=dtype).view(-1, 1, 1)
            noisy = (1.0 - s) * packed + s * noise
            store.enabled = True
            store.reset()
            with torch.no_grad():
                transformer(
                    hidden_states=noisy,
                    timestep=torch.tensor([sig], device=device, dtype=dtype),
                    guidance=guidance,
                    pooled_projections=pooled,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )
            store.enabled = False
            agg = store.aggregate(h, w)[0]  # (H, W, n_tokens)

            per_word = []
            for idxs in token_idxs:
                m = agg[:, :, idxs].mean(dim=-1).unsqueeze(0).unsqueeze(0).float()
                per_word.append(normalize_map(smoothing(m))[0, 0].cpu().numpy())
            maps_by_sigma[sig] = per_word

            p0, p1 = peak_xy(per_word[0]), peak_xy(per_word[1])
            diag = float(np.hypot(h, w))
            acc[sig]["peak_sep"].append(float(np.hypot(p0[0] - p1[0], p0[1] - p1[1])) / diag)
            acc[sig]["iou"].append(mask_iou(per_word[0], per_word[1]))

            corrs = []
            for m in per_word:
                mz = (m - m.mean()) / (m.std() + 1e-8)
                corrs.append(float((mz * energy).mean()))
            acc[sig]["vae_corr"].append(float(np.mean(corrs)))

        ref = maps_by_sigma[min(args.sigmas)]
        for sig in args.sigmas:
            sims = []
            for a, b in zip(maps_by_sigma[sig], ref):
                az = (a - a.mean()) / (a.std() + 1e-8)
                bz = (b - b.mean()) / (b.std() + 1e-8)
                sims.append(float((az * bz).mean()))
            acc[sig]["consistency"].append(float(np.mean(sims)))

        if saved < args.save_maps:
            _save_grid(maps_by_sigma, args.sigmas, words, prompt, args, saved)
            saved += 1

        if si % 10 == 0:
            print(f"  [{si}/{n}] {prompt!r} nouns={words}", flush=True)

    rows = []
    for sig in args.sigmas:
        d = acc[sig]
        if not d["peak_sep"]:
            continue
        rows.append(
            {
                "sigma": sig,
                "n": len(d["peak_sep"]),
                "peak_sep": round(float(np.mean(d["peak_sep"])), 4),
                "iou_overlap": round(float(np.mean(d["iou"])), 4),
                "vae_corr": round(float(np.mean(d["vae_corr"])), 4),
                "consistency_vs_min_sigma": round(float(np.mean(d["consistency"])), 4),
            }
        )

    path = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(path, "w") as f:
        json.dump({"tag": args.tag, "lora": args.lora, "n_samples": n, "rows": rows}, f, indent=2)

    print("\nsigma   peak_sep  iou_overlap  vae_corr  consistency")
    for r in rows:
        print(
            f"{r['sigma']:.2f}    {r['peak_sep']:.4f}    {r['iou_overlap']:.4f}"
            f"       {r['vae_corr']:+.4f}   {r['consistency_vs_min_sigma']:+.4f}"
        )
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
