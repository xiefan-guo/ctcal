"""Shared paths, defaults and hyperparameters.

Everything machine-specific is read from the environment:

  CTCAL_FLUX_PATH       a local directory holding FLUX.1-dev (optional)
  CTCAL_HF_HOME         HF cache root; also searched for the Stanza resources
  HF_HOME               standard Hugging Face cache root (used if the above is unset)
  HF_ENDPOINT           Hub mirror to resolve from
  STANZA_RESOURCES_DIR  Stanza model directory (optional)

With neither CTCAL_FLUX_PATH nor an HF cache root set, the FLUX repo id is passed
straight to `from_pretrained`, which downloads into the default Hub cache.
"""

from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FLUX_REPO = "black-forest-labs/FLUX.1-dev"


def hf_home() -> str | None:
    """Root of the Hugging Face cache, or None to use huggingface_hub's default."""
    return os.environ.get("CTCAL_HF_HOME") or os.environ.get("HF_HOME") or None


def hf_hub_cache() -> str | None:
    """The `hub/` subdirectory of the cache, where repo snapshots are stored."""
    root = hf_home()
    return os.path.join(root, "hub") if root else None


def stanza_dir() -> str | None:
    """Where the Stanza English models live, or None to let Stanza decide."""
    explicit = os.environ.get("STANZA_RESOURCES_DIR")
    if explicit:
        return explicit
    root = hf_home()
    return os.path.join(root, "stanza_resources") if root else None


# Every file a FluxPipeline load touches, including all weight shards.
_FLUX_REQUIRED = [
    "model_index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "text_encoder/model.safetensors",
    "tokenizer/vocab.json",
    "tokenizer_2/spiece.model",
    "tokenizer_2/tokenizer_config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "scheduler/scheduler_config.json",
] + [
    f"transformer/diffusion_pytorch_model-{i:05d}-of-00003.safetensors" for i in (1, 2, 3)
] + [
    f"text_encoder_2/model-{i:05d}-of-00002.safetensors" for i in (1, 2)
]


def is_complete_flux(root: str) -> bool:
    """Whether `root` holds a full FLUX.1-dev snapshot.

    A partial copy is worse than no copy: an unfinished tokenizer_2/spiece.model
    silently tokenises every word to <unk>, which would disable CTCal's token
    alignment without raising.
    """
    return all(os.path.exists(os.path.join(root, f)) for f in _FLUX_REQUIRED)


def flux_path() -> str:
    """The FLUX weights to load: a verifiably complete local copy, or the repo id."""
    override = os.environ.get("CTCAL_FLUX_PATH")
    if override:
        if not is_complete_flux(override):
            raise RuntimeError(
                f"CTCAL_FLUX_PATH={override} is missing required FLUX files; "
                "finish staging it or unset the variable to load from the Hub."
            )
        return override
    return FLUX_REPO


PROMPTS_DIR = os.path.join(REPO_ROOT, "data", "prompts")

BLIP_VQA = "Salesforce/blip-vqa-capfilt-large"
CLIP_MODEL = "openai/clip-vit-large-patch14"

# --- CTCal -------------------------------------------------------------------
# FLUX.1-dev is guidance-distilled and takes a guidance *embedding* rather than
# classifier-free guidance; 3.5 is the value its own pipeline uses.
DEFAULT_GUIDANCE = 3.5

# The middle double-stream blocks, where text-token semantics are strongest.
DEFAULT_ATTN_LAYERS = [6, 8, 10, 12, 14, 16, 18]

# Loss weights (Eq. 7). lambda1 alone is on by default: on FLUX the semantic term
# decays to ~1e-6 as the autoencoder trains, and with it off the reconstruction
# proxy has nothing to train. All four stay here so Eq. 7 is implemented in full.
LAMBDA1 = 0.01
LAMBDA2 = 0.0
LAMBDA3 = 0.0
LAMBDA4 = 0.0
TAU = 0.05

TRAIN_RESOLUTION = 512


def attn_resolution(resolution: int = TRAIN_RESOLUTION) -> int:
    """Patch-grid side length of the aggregated attention maps.

    FLUX's VAE downsamples 8x and its transformer packs the latent 2x2, so each 2x2
    latent block becomes one image token: the grid is resolution / 16. This is also
    the input resolution the attention autoencoder is built for.
    """
    return resolution // 16


ATTN_RESOLUTION = attn_resolution()

# The supplementary uses kernel 5 / sigma 0.5 for SD 3's 64x64 maps; our grid is
# 32x32, between SD 2.1's 16x16 and SD 3's 64x64, so kernel 3 keeps the blur's
# spatial extent comparable.
SMOOTH_KERNEL = 3
SMOOTH_SIGMA = 0.5

T5_MAX_LENGTH = 128  # FLUX.1-dev supports 512; compositional prompts are short.
