"""One end-to-end CTCal training step on real FLUX weights.

Verifies the parts that only break when wired together: both passes yield A_IT at
the expected patch resolution, L_CTCal produces gradients on the LoRA parameters but
not on the frozen backbone, an optimiser step changes those weights, and the
flow-matching target reconstructs the clean latent.
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctcal import config  # noqa: E402
from ctcal.attention import install_attention_store  # noqa: E402
from ctcal.loss import CTCalLoss  # noqa: E402
from ctcal.modules import AttentionAutoencoder, GaussianSmoothing  # noqa: E402
from ctcal.pos_select import get_nlp, select_words, word_to_token_indices  # noqa: E402
from ctcal.timesteps import lambda_t, sample_student_sigmas  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


def main():
    from diffusers import FluxPipeline
    from peft import LoraConfig

    torch.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16
    res = 512
    attn_res = res // 16

    print(f"loading FLUX from {config.flux_path()} ...", flush=True)
    pipe = FluxPipeline.from_pretrained(config.flux_path(), dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    transformer, vae = pipe.transformer, pipe.vae

    for m in (transformer, vae, pipe.text_encoder, pipe.text_encoder_2):
        m.requires_grad_(False)

    transformer.add_adapter(
        LoraConfig(
            r=8,
            lora_alpha=8,
            init_lora_weights="gaussian",
            target_modules=[
                "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
            ],
        )
    )
    lora_params = []
    for name, p in transformer.named_parameters():
        if "lora" in name:
            p.data = p.data.float()
            p.requires_grad_(True)
            lora_params.append(p)
    check("LoRA params exist", len(lora_params) > 0, f"{len(lora_params)} tensors")

    layers = config.DEFAULT_ATTN_LAYERS
    store = install_attention_store(transformer, layers)
    ae = AttentionAutoencoder(resolution=attn_res).to(device).float()
    sm = GaussianSmoothing(config.SMOOTH_KERNEL, config.SMOOTH_SIGMA).to(device)
    # config leaves lambda2/3/4 at 0; this end-to-end step exercises every branch.
    loss_fn = CTCalLoss(
        ae, sm, config.LAMBDA1, lambda2=1.0, lambda3=0.01, lambda4=0.1, tau=config.TAU
    )
    nlp = get_nlp(config.stanza_dir())

    prompts = ["a red car and a yellow clock", "a green backpack and a blue banana"]
    bsz = len(prompts)

    # Stand-in for real training images.
    pixel_values = torch.rand(bsz, 3, res, res, device=device, dtype=dtype) * 2 - 1
    reward = torch.tensor([0.9, 0.7], device=device)

    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
        prompt_embeds, pooled, text_ids = pipe.encode_prompt(
            prompt=prompts, prompt_2=prompts, device=device,
            num_images_per_prompt=1, max_sequence_length=config.T5_MAX_LENGTH,
        )
    lh, lw = latents.shape[2], latents.shape[3]
    h, w = lh // 2, lw // 2
    check("patch grid matches config", (h, w) == (attn_res, attn_res), f"{(h, w)} vs {attn_res}")

    packed = pipe._pack_latents(latents, bsz, latents.shape[1], lh, lw)
    img_ids = pipe._prepare_latent_image_ids(bsz, h, w, device, dtype)
    guidance = torch.full((bsz,), config.DEFAULT_GUIDANCE, device=device, dtype=dtype)

    noise = torch.randn_like(packed)
    sigmas = sample_student_sigmas(bsz, shift=3.0, device=device)
    tea_sigma = 0.15
    print(f"  sigmas(student)={[round(x, 4) for x in sigmas.tolist()]} teacher={tea_sigma:.4f}")

    s = sigmas.view(-1, 1, 1).to(dtype)
    noisy = (1.0 - s) * packed + s * noise
    target = noise - packed

    # x_t - t*v = (1-t)x_0 + t*e - t*(e - x_0) = x_0, so subtracting the velocity
    # target must recover the clean latent.
    recon = noisy - s * target
    err = (recon.float() - packed.float()).abs().max().item()
    check("flow parameterisation recovers x_0", err < 0.05, f"max err {err:.4f}")

    # --- student pass -------------------------------------------------------
    store.enabled, store.detach = True, False
    store.reset()
    pred = transformer(
        hidden_states=noisy, timestep=sigmas.to(dtype), guidance=guidance,
        pooled_projections=pooled, encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids, img_ids=img_ids, return_dict=False,
    )[0]
    check("maps collected from every hooked layer", len(store.maps) == len(layers), f"{len(store.maps)}")
    attn_stu = store.aggregate(h, w)
    check(
        "A_stu shape",
        attn_stu.shape[:3] == (bsz, h, w),
        f"{tuple(attn_stu.shape)}",
    )
    check("A_stu requires grad", attn_stu.requires_grad)

    loss_diff = (
        reward * F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=[1, 2])
    ).mean()
    check("flow loss finite", torch.isfinite(loss_diff), f"{loss_diff.item():.4f}")

    # --- teacher pass -------------------------------------------------------
    tea_sigmas = torch.full_like(sigmas, tea_sigma)
    tea_sigmas = torch.where(tea_sigmas < sigmas, tea_sigmas, sigmas * 0.5)
    s_t = tea_sigmas.view(-1, 1, 1).to(dtype)
    noisy_tea = (1.0 - s_t) * packed + s_t * noise

    store.enabled, store.detach = True, True
    store.reset()
    with torch.no_grad():
        transformer(
            hidden_states=noisy_tea, timestep=tea_sigmas.to(dtype), guidance=guidance,
            pooled_projections=pooled, encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids, img_ids=img_ids, return_dict=False,
        )
    attn_tea = store.aggregate(h, w)
    store.enabled = False
    check("A_tea shape matches A_stu", attn_tea.shape == attn_stu.shape)
    check("A_tea detached", not attn_tea.requires_grad)
    check("t_tea < t_stu for all samples", bool((tea_sigmas < sigmas).all()))

    # --- token selection ----------------------------------------------------
    token_idxs, valid_lens = [], []
    for prompt in prompts:
        enc = pipe.tokenizer_2(
            prompt, padding="max_length", max_length=config.T5_MAX_LENGTH, truncation=True
        )
        valid_lens.append(int(sum(enc["attention_mask"])))
        words = select_words(prompt, nlp)
        token_idxs.append(word_to_token_indices(prompt, words, pipe.tokenizer_2, config.T5_MAX_LENGTH))
    print(f"  token_idxs={token_idxs} valid_lens={valid_lens}")
    check("two nouns selected per prompt", all(len([t for t in ti if t]) == 2 for ti in token_idxs))

    # --- CTCal loss and backward -------------------------------------------
    l_ctcal, logs = loss_fn(attn_stu, attn_tea, token_idxs, lambda_t(sigmas), valid_lens=valid_lens)
    print(f"  L_CTCal={l_ctcal.item():.6f} components={ {k: round(v, 6) for k, v in logs.items()} }")
    check("L_CTCal finite and non-zero", torch.isfinite(l_ctcal) and l_ctcal.item() > 0)
    check("pixel term non-zero", logs["pixel"] > 0, f"{logs['pixel']:.6f}")
    check("semantic term non-zero", logs["semantic"] > 0, f"{logs['semantic']:.6f}")
    check("recon term non-zero", logs["recon"] > 0, f"{logs['recon']:.6f}")

    before = [p.detach().clone() for p in lora_params]
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": 1e-4}, {"params": list(ae.parameters()), "lr": 1e-4}]
    )
    (loss_diff + l_ctcal).backward()

    n_grad = sum(1 for p in lora_params if p.grad is not None and p.grad.abs().sum() > 0)
    check("gradients reach LoRA params", n_grad > 0, f"{n_grad}/{len(lora_params)} tensors")
    ae_grad = sum(1 for p in ae.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    check("gradients reach autoencoder", ae_grad > 0, f"{ae_grad} tensors")
    check(
        "frozen backbone has no grads",
        all(p.grad is None for n, p in transformer.named_parameters() if "lora" not in n),
    )

    opt.step()
    moved = sum(1 for b, p in zip(before, lora_params) if not torch.equal(b, p.detach()))
    check("optimiser step changes LoRA weights", moved > 0, f"{moved}/{len(lora_params)} tensors")

    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"\npeak GPU memory: {mem:.1f} GB")

    print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURES: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
