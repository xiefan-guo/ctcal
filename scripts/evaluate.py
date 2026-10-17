"""Generate images for a benchmark split and score them.

Reports the T2I-CompBench++ BLIP-VQA score for the attribute-binding categories plus
CLIPScore. Pass `--lora` to evaluate a fine-tuned checkpoint; omit it for the
pre-trained FLUX baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctcal import config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--lora", default=None, help="directory holding pytorch_lora_weights.safetensors")
    p.add_argument("--tag", default="baseline", help="label for the output files")
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--images-per-prompt", type=int, default=2)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=config.DEFAULT_GUIDANCE)
    p.add_argument("--resolution", type=int, default=config.TRAIN_RESOLUTION)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out-dir", default=os.path.join(config.REPO_ROOT, "results"))
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--no-clip", action="store_true", help="skip CLIPScore")
    return p.parse_args()


def main():
    args = parse_args()
    from diffusers import FluxPipeline

    from ctcal.scorer import BlipVQAScorer, ClipScorer

    prompt_file = os.path.join(config.PROMPTS_DIR, f"{args.category}_{args.split}.txt")
    with open(prompt_file) as f:
        prompts = [ln.strip() for ln in f if ln.strip()][: args.num_prompts]
    print(f"{args.category}/{args.split}: {len(prompts)} prompts, tag={args.tag}", flush=True)

    pipe = FluxPipeline.from_pretrained(config.flux_path(), dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    if args.lora:
        pipe.load_lora_weights(args.lora)
        print(f"loaded LoRA from {args.lora}", flush=True)

    hub = config.hf_hub_cache()
    blip = BlipVQAScorer(config.BLIP_VQA, device="cuda", cache_dir=hub)
    clip = None if args.no_clip else ClipScorer(config.CLIP_MODEL, device="cuda", cache_dir=hub)

    out_root = os.path.join(args.out_dir, f"{args.category}_{args.split}_{args.tag}")
    os.makedirs(out_root, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(out_root, "images"), exist_ok=True)

    records = []
    # Fixed seeds per (prompt, repeat) so every method is compared on the same noise.
    jobs = [
        (pi, prompt, r, args.seed + 7919 * pi + r)
        for pi, prompt in enumerate(prompts)
        for r in range(args.images_per_prompt)
    ]

    for start in range(0, len(jobs), args.batch_size):
        chunk = jobs[start : start + args.batch_size]
        gens = [torch.Generator("cuda").manual_seed(j[3]) for j in chunk]
        out = pipe(
            prompt=[j[1] for j in chunk],
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=gens,
        )
        images = out.images
        texts = [j[1] for j in chunk]
        b_scores = blip.score_batch(images, texts)
        c_scores = clip.score_batch(images, texts) if clip else [float("nan")] * len(images)

        for (pi, prompt, r, seed), img, bs, cs in zip(chunk, images, b_scores, c_scores):
            if args.save_images:
                img.save(os.path.join(out_root, "images", f"p{pi:04d}_r{r}.jpg"), quality=95)
            records.append({"prompt": prompt, "repeat": r, "seed": seed, "blip_vqa": bs, "clip": cs})

        if start % (args.batch_size * 10) == 0:
            print(
                f"  [{len(records)}/{len(jobs)}] running BLIP-VQA="
                f"{statistics.mean(x['blip_vqa'] for x in records):.4f}",
                flush=True,
            )

    summary = {
        "category": args.category,
        "split": args.split,
        "tag": args.tag,
        "lora": args.lora,
        "n_prompts": len(prompts),
        "n_images": len(records),
        "blip_vqa": round(statistics.mean(r["blip_vqa"] for r in records), 4),
        "clip": round(statistics.mean(r["clip"] for r in records), 4),
    }
    with open(os.path.join(out_root, "per_image.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
