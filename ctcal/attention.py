"""Cross-attention (A_IT) extraction for FLUX MM-DiT blocks.

FLUX concatenates text and image tokens into one sequence and runs a joint
self-attention over it, so the joint map decomposes into blocks

    A = [[A_II, A_IT],
         [A_TI, A_TT]]

CTCal only needs A_IT: for every image patch, how much it attends to each text
token. The stock `FluxAttnProcessor` calls a fused SDPA kernel that never
materialises the probability matrix, so we install a processor that recomputes the
image-query x text-key block explicitly and stores it.

Only the double-stream blocks carry a separate `encoder_hidden_states` stream with
its own text projections, so those are the blocks we hook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import _get_qkv_projections


class AttentionStore:
    """Collects A_IT blocks from the hooked FLUX double-stream blocks."""

    def __init__(self):
        self.maps: list[torch.Tensor] = []
        self.enabled = False
        # False keeps the graph (student pass), True saves memory (teacher pass).
        self.detach = False

    def reset(self):
        self.maps = []

    def store(self, attn_probs: torch.Tensor):
        if not self.enabled:
            return
        self.maps.append(attn_probs.detach() if self.detach else attn_probs)

    def aggregate(self, height: int, width: int) -> torch.Tensor:
        """Average the stored A_IT over layers and heads.

        Returns (B, height, width, n_text_tokens).
        """
        if not self.maps:
            raise RuntimeError("AttentionStore is empty - was `enabled` set before the forward pass?")
        # Each entry is (B, heads, img_len, txt_len); mean over heads then over layers.
        stacked = torch.stack([m.mean(dim=1) for m in self.maps], dim=0).mean(dim=0)
        b, img_len, txt_len = stacked.shape
        if img_len != height * width:
            raise ValueError(f"image token count {img_len} != {height}*{width}")
        return stacked.view(b, height, width, txt_len)


class CTCalFluxAttnProcessor:
    """FLUX joint-attention processor that also exposes the A_IT block.

    The output path is numerically identical to `FluxAttnProcessor`; the extra work
    is a second, narrower attention computation (image queries x text keys only)
    used as the CTCal supervision signal.
    """

    def __init__(self, store: AttentionStore, layer_idx: int):
        self.store = store
        self.layer_idx = layer_idx

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (-1, attn.head_dim))
        key = key.unflatten(-1, (-1, attn.head_dim))
        value = value.unflatten(-1, (-1, attn.head_dim))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        txt_len = 0
        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (-1, attn.head_dim))
            encoder_key = encoder_key.unflatten(-1, (-1, attn.head_dim))
            encoder_value = encoder_value.unflatten(-1, (-1, attn.head_dim))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            txt_len = encoder_query.shape[1]
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        if self.store.enabled and txt_len > 0:
            # A_IT only: image queries against text keys. Softmax must run over the
            # full joint key axis to match the real attention distribution, so we
            # compute all logits and keep just the text columns.
            scale = attn.head_dim**-0.5
            logits = torch.matmul(q * scale, k.transpose(-1, -2))
            probs = logits.softmax(dim=-1)
            # rows = image tokens, cols = text tokens
            self.store.store(probs[:, :, txt_len:, :txt_len])
            del logits, probs

        hidden_states = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [txt_len, hidden_states.shape[1] - txt_len], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states.contiguous())
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states
        return hidden_states


def install_attention_store(transformer, layers: list[int]) -> AttentionStore:
    """Swap in CTCal processors on the given double-stream block indices."""
    store = AttentionStore()
    blocks = transformer.transformer_blocks
    for idx in layers:
        if not 0 <= idx < len(blocks):
            raise ValueError(f"double-stream block index {idx} out of range (0..{len(blocks) - 1})")
        blocks[idx].attn.processor = CTCalFluxAttnProcessor(store, idx)
    return store
