"""Build the CTCal training set by reward-driven sample selection (GORS protocol).

Paper Sec. 4.1: for each prompt, generate k candidate images with the target model,
score each with the benchmark's own metric, and keep the top-scoring pairs. The
resulting text-image set is what both the GORS baseline and CTCal fine-tune on, so
the only difference between them is the loss.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctcal import config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True, help="e.g. color, shape, texture")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--k", type=int, default=8, help="candidates generated per prompt")
    p.add_argument("--top-n", type=int, default=2, help="candidates kept per prompt")
    p.add_argument("--num-prompts", type=int, default=None, help="cap on prompts (debug)")
    p.add_argument("--steps", type=int, default=20, help="inference steps per candidate")
    p.add_argument("--guidance", type=float, default=config.DEFAULT_GUIDANCE)
    p.add_argument("--resolution", type=int, default=config.TRAIN_RESOLUTION)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out-dir", default=os.path.join(config.REPO_ROOT, "data", "train"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", type=int, default=0, help="this worker's index")
    p.add_argument("--num-shards", type=int, default=1, help="total workers")
    return p.parse_args()


def main():
    args = parse_args()
    from diffusers import FluxPipeline

    from ctcal.scorer import BlipVQAScorer

    prompt_file = os.path.join(config.PROMPTS_DIR, f"{args.category}_{args.split}.txt")
    with open(prompt_file) as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    if args.num_prompts:
        prompts = prompts[: args.num_prompts]
    # Shard by stride so every worker sees a similar mix of prompt templates.
    prompts = prompts[args.shard :: args.num_shards]

    out_root = os.path.join(args.out_dir, args.category)
    img_dir = os.path.join(out_root, "images")
    os.makedirs(img_dir, exist_ok=True)
    meta_path = os.path.join(out_root, f"metadata.shard{args.shard}.jsonl")

    # Resume: skip prompts this shard already recorded.
    done = set()
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["prompt"])
                except (json.JSONDecodeError, KeyError):
                    continue
    todo = [p for p in prompts if p not in done]
    print(
        f"[shard {args.shard}/{args.num_shards}] {args.category}/{args.split}: "
        f"{len(todo)} prompts to do ({len(done)} already done)",
        flush=True,
    )
    if not todo:
        return

    pipe = FluxPipeline.from_pretrained(config.flux_path(), dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    scorer = BlipVQAScorer(config.BLIP_VQA, device="cuda", cache_dir=config.hf_hub_cache())

    meta_file = open(meta_path, "a")
    for pi, prompt in enumerate(todo):
        images, seeds = [], []
        for start in range(0, args.k, args.batch_size):
            n = min(args.batch_size, args.k - start)
            batch_seeds = [args.seed + 100000 * args.shard + 997 * pi + start + j for j in range(n)]
            gens = [torch.Generator("cuda").manual_seed(s) for s in batch_seeds]
            out = pipe(
                prompt=[prompt] * n,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=gens,
            )
            images.extend(out.images)
            seeds.extend(batch_seeds)

        scores = scorer.score_batch(images, [prompt] * len(images))
        order = sorted(range(len(images)), key=lambda i: -scores[i])[: args.top_n]

        for rank, idx in enumerate(order):
            slug = prompt.replace(" ", "_").replace("/", "-")[:80]
            fname = f"{args.category}_s{args.shard}_p{pi:05d}_r{rank}_{slug}.jpg"
            images[idx].save(os.path.join(img_dir, fname), quality=95)
            meta_file.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "image": fname,
                        "reward": scores[idx],
                        "seed": seeds[idx],
                        "category": args.category,
                    }
                )
                + "\n"
            )
        meta_file.flush()

        if pi % 10 == 0:
            best = max(scores)
            print(
                f"  [{pi}/{len(todo)}] {prompt!r} best={best:.4f} "
                f"mean={sum(scores) / len(scores):.4f}",
                flush=True,
            )
    meta_file.close()
    print(f"[shard {args.shard}] done -> {meta_path}", flush=True)


if __name__ == "__main__":
    main()
