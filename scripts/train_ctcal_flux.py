"""Fine-tune FLUX.1-dev with CTCal, or with the GORS baseline.

One optimisation step:

  1. encode the image to latents and the prompt to T5/CLIP embeddings;
  2. sample t_stu from FLUX's native logit-normal schedule and run the student pass
     at that noise level, keeping the graph -- this pass carries both the
     flow-matching loss and A_stu;
  3. run the teacher pass at the smaller t_tea under no_grad to get A_tea;
  4. add L_CTCal(A_stu, A_tea) to the flow-matching loss, weighted by
     lambda_t = t_stu / T_train.

The two passes share one set of LoRA-adapted weights: A_tea is not produced by a
frozen copy, so it improves as training proceeds.

`--mode gors` skips steps 3-4 entirely and trains on the reward-weighted
flow-matching loss alone.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctcal import config  # noqa: E402
from ctcal.attention import install_attention_store  # noqa: E402
from ctcal.data import RewardSelectedDataset, collate  # noqa: E402
from ctcal.loss import CTCalLoss  # noqa: E402
from ctcal.modules import AttentionAutoencoder, GaussianSmoothing  # noqa: E402
from ctcal.pos_select import get_nlp, select_words, word_to_token_indices  # noqa: E402
from ctcal.timesteps import (  # noqa: E402
    lambda_t,
    sample_student_sigmas,
    teacher_sigma_from_diagnostic,
)

DEFAULT_TEACHER_SIGMA = 0.15


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="ctcal", choices=["ctcal", "gors"])
    p.add_argument("--data-roots", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resolution", type=int, default=config.TRAIN_RESOLUTION)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="counts *samples*, so a run has max-steps / grad-accum optimiser updates",
    )
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--ae-lr", type=float, default=5e-5)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--attn-layers", type=int, nargs="+", default=config.DEFAULT_ATTN_LAYERS)
    p.add_argument(
        "--teacher-sigma",
        type=float,
        default=None,
        help="fixed t_tea; default reads the diagnostic, else "
        f"{DEFAULT_TEACHER_SIGMA}",
    )
    p.add_argument("--no-path-check", action="store_true", help="skip verifying every image path")
    return p.parse_args()


def encode_prompts(pipe, prompts, device, max_length):
    """T5 sequence embeddings + pooled CLIP embedding, as FLUX expects."""
    prompt_embeds, pooled, text_ids = pipe.encode_prompt(
        prompt=prompts,
        prompt_2=prompts,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_length,
    )
    return prompt_embeds, pooled, text_ids


def build_optimizer(lora_params, autoencoder, args):
    param_groups = [{"params": lora_params, "lr": args.lr}]
    if autoencoder is not None:
        param_groups.append({"params": list(autoencoder.parameters()), "lr": args.ae_lr})
    return torch.optim.AdamW(
        param_groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay
    )


def build_scheduler(optimizer, args):
    """Optional warmup + cosine decay to 0, applied to every param group equally."""
    if args.lr_schedule != "cosine":
        return None
    total_updates = max(1, args.max_steps // args.grad_accum)
    warmup = max(1, int(args.warmup_frac * total_updates))

    def _lr_lambda(update: int) -> float:
        if update < warmup:
            return (update + 1) / warmup
        prog = (update - warmup) / max(1, total_updates - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    print(f"lr schedule: cosine, peak {args.lr}, warmup {warmup}/{total_updates} updates",
          flush=True)
    return scheduler


def resolve_teacher_sigma(args) -> float:
    """t_tea: explicit, else the diagnostic's measurement, else a fixed fallback."""
    if args.teacher_sigma is not None:
        return args.teacher_sigma
    diag = os.path.join(config.REPO_ROOT, "results", "diagnose", "pretrained.json")
    measured = teacher_sigma_from_diagnostic(diag)
    if measured is not None:
        print(f"teacher sigma {measured:.4f} (from {diag})", flush=True)
        return measured
    print(f"teacher sigma {DEFAULT_TEACHER_SIGMA} (no diagnostic found)", flush=True)
    return DEFAULT_TEACHER_SIGMA


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda"
    dtype = torch.bfloat16

    from diffusers import FluxPipeline
    from peft import LoraConfig, get_peft_model_state_dict

    print("loading FLUX...", flush=True)
    pipe = FluxPipeline.from_pretrained(config.flux_path(), dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    transformer, vae = pipe.transformer, pipe.vae
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    vae.eval()
    pipe.text_encoder.eval()
    pipe.text_encoder_2.eval()

    # LoRA on the attention projections of the denoising network, where FLUX's joint
    # attention lives. The text stream is a frozen 4.7B T5 and is not adapted.
    transformer.add_adapter(
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=[
                "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
            ],
        )
    )
    # LoRA params must be fp32 for stable optimisation.
    lora_params = []
    for name, param in transformer.named_parameters():
        if "lora" in name:
            param.data = param.data.float()
            param.requires_grad_(True)
            lora_params.append(param)
    print(f"LoRA params: {sum(p.numel() for p in lora_params) / 1e6:.2f}M", flush=True)

    use_ctcal = args.mode == "ctcal"
    attn_res = config.attn_resolution(args.resolution)

    store = autoencoder = ctcal_loss = nlp = teacher_sigma = None
    if use_ctcal:
        store = install_attention_store(transformer, args.attn_layers)
        autoencoder = AttentionAutoencoder(resolution=attn_res).to(device).float()
        smoothing = GaussianSmoothing(config.SMOOTH_KERNEL, config.SMOOTH_SIGMA).to(device)
        ctcal_loss = CTCalLoss(
            autoencoder=autoencoder,
            smoothing=smoothing,
            lambda1=config.LAMBDA1,
            lambda2=config.LAMBDA2,
            lambda3=config.LAMBDA3,
            lambda4=config.LAMBDA4,
            tau=config.TAU,
        )
        nlp = get_nlp(config.stanza_dir())
        teacher_sigma = resolve_teacher_sigma(args)
        print(
            f"CTCal: layers={args.attn_layers} attn_res={attn_res} "
            f"teacher_sigma={teacher_sigma:.4f} "
            f"lambdas=({config.LAMBDA1}, {config.LAMBDA2}, {config.LAMBDA3}, {config.LAMBDA4})",
            flush=True,
        )

    optimizer = build_optimizer(lora_params, autoencoder, args)
    scheduler = build_scheduler(optimizer, args)

    dataset = RewardSelectedDataset(
        args.data_roots, resolution=args.resolution, verify_paths=not args.no_path_check
    )
    print(f"dataset: {len(dataset)} pairs", flush=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate,
        drop_last=True,
        persistent_workers=True,
    )

    vae_scale = vae.config.scaling_factor
    vae_shift = vae.config.shift_factor
    log_file = open(os.path.join(args.output_dir, "train_log.jsonl"), "a")

    step = 0
    t_start = time.time()
    accum_loss = accum_diff = accum_ctcal = 0.0
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            prompts = batch["prompt"]
            bsz = len(prompts)
            reward = batch["reward"].to(device)

            with torch.no_grad():
                pixel_values = batch["pixel_values"].to(device, dtype=dtype)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = (latents - vae_shift) * vae_scale
                prompt_embeds, pooled, text_ids = encode_prompts(
                    pipe, prompts, device, config.T5_MAX_LENGTH
                )

            lh, lw = latents.shape[2], latents.shape[3]
            packed = pipe._pack_latents(latents, bsz, latents.shape[1], lh, lw)
            img_ids = pipe._prepare_latent_image_ids(bsz, lh // 2, lw // 2, device, dtype)
            grid = (lh // 2, lw // 2)

            noise = torch.randn_like(packed)
            sigmas = sample_student_sigmas(bsz, shift=3.0, device=device)

            # Rectified flow: x_t = (1 - t) * x_0 + t * noise, target velocity = noise - x_0.
            s = sigmas.view(-1, 1, 1).to(dtype)
            noisy = (1.0 - s) * packed + s * noise
            target = noise - packed
            guidance = torch.full((bsz,), config.DEFAULT_GUIDANCE, device=device, dtype=dtype)

            if use_ctcal:
                store.enabled = True
                store.detach = False
                store.reset()

            model_pred = transformer(
                hidden_states=noisy,
                timestep=sigmas.to(dtype),
                guidance=guidance,
                pooled_projections=pooled,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]

            # Reward-weighted flow-matching loss (GORS, supplementary Eq. 1).
            loss_diff = (
                reward
                * F.mse_loss(model_pred.float(), target.float(), reduction="none").mean(
                    dim=list(range(1, model_pred.ndim))
                )
            ).mean()

            loss = loss_diff
            ctcal_logs = {}

            if use_ctcal:
                attn_stu = store.aggregate(*grid)

                tea_sigmas = torch.full_like(sigmas, teacher_sigma)
                # t_tea must stay below t_stu; where it would not, fall back to half
                # of t_stu so the pair still respects the ordering.
                tea_sigmas = torch.where(tea_sigmas < sigmas, tea_sigmas, sigmas * 0.5)

                s_tea = tea_sigmas.view(-1, 1, 1).to(dtype)
                noisy_tea = (1.0 - s_tea) * packed + s_tea * noise

                store.enabled = True
                store.detach = True
                store.reset()
                with torch.no_grad():
                    transformer(
                        hidden_states=noisy_tea,
                        timestep=tea_sigmas.to(dtype),
                        guidance=guidance,
                        pooled_projections=pooled,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=img_ids,
                        return_dict=False,
                    )
                attn_tea = store.aggregate(*grid)
                store.enabled = False

                token_idxs, valid_lens = [], []
                for prompt in prompts:
                    enc = pipe.tokenizer_2(
                        prompt,
                        padding="max_length",
                        max_length=config.T5_MAX_LENGTH,
                        truncation=True,
                    )
                    valid_lens.append(int(sum(enc["attention_mask"])))
                    words = select_words(prompt, nlp)
                    token_idxs.append(
                        word_to_token_indices(
                            prompt, words, pipe.tokenizer_2, config.T5_MAX_LENGTH
                        )
                    )

                l_ctcal, ctcal_logs = ctcal_loss(
                    attn_stu, attn_tea, token_idxs, lambda_t(sigmas), valid_lens=valid_lens
                )
                loss = loss + l_ctcal
                accum_ctcal += float(l_ctcal.detach()) / args.grad_accum

            accum_loss += float(loss.detach()) / args.grad_accum
            accum_diff += float(loss_diff.detach()) / args.grad_accum

            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1

            if step % args.log_every == 0:
                rec = {
                    "step": step,
                    "loss": round(accum_loss * args.grad_accum / args.log_every, 6),
                    "loss_diffusion": round(accum_diff * args.grad_accum / args.log_every, 6),
                    "loss_ctcal": round(accum_ctcal * args.grad_accum / args.log_every, 6),
                    "sigma_mean": round(float(sigmas.mean()), 4),
                    "elapsed_s": round(time.time() - t_start, 1),
                    **{f"ctcal_{k}": round(v, 6) for k, v in ctcal_logs.items()},
                }
                print(json.dumps(rec), flush=True)
                log_file.write(json.dumps(rec) + "\n")
                log_file.flush()
                accum_loss = accum_diff = accum_ctcal = 0.0

            if step % args.save_every == 0 or step == args.max_steps:
                ckpt = os.path.join(args.output_dir, f"checkpoint-{step}")
                os.makedirs(ckpt, exist_ok=True)
                lora_sd = get_peft_model_state_dict(transformer)
                FluxPipeline.save_lora_weights(ckpt, lora_sd, safe_serialization=True)
                if autoencoder is not None:
                    torch.save(autoencoder.state_dict(), os.path.join(ckpt, "attn_autoencoder.pt"))
                print(f"saved {ckpt}", flush=True)

    log_file.close()
    print(f"done in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
