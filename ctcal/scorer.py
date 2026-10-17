"""BLIP-VQA and CLIP scorers.

BLIP-VQA is T2I-CompBench++'s metric for the attribute-binding categories. The
protocol: split a compositional prompt into its noun phrases, ask BLIP a yes/no
question per phrase, and take the product of P("yes") across phrases. A product
(rather than a mean) makes the metric compositional -- getting one object right
while missing the other scores near zero.

The same score is the reward for GORS-style dataset construction, so training data
selection and evaluation share this code.
"""

from __future__ import annotations

import re

import torch
from PIL import Image


def split_noun_phrases(prompt: str) -> list[str]:
    """Split a T2I-CompBench prompt into the phrases to ask about.

    The benchmark's attribute prompts are built from templates joined by "and",
    e.g. "a green bench and a red car" -> ["a green bench", "a red car"]. One
    question per half, mirroring the official BLIP-VQA evaluator.
    """
    text = prompt.strip().rstrip(".")
    parts = re.split(r"\s+and\s+", text)
    return [p.strip() for p in parts if p.strip()]


class BlipVQAScorer:
    """Scores text-image alignment as prod_i P("yes" | "{phrase}?")."""

    def __init__(
        self,
        model_name: str,
        device="cuda",
        dtype=torch.float16,
        cache_dir=None,
        normalize: bool = True,
    ):
        from transformers import BlipForQuestionAnswering, BlipProcessor

        self.processor = BlipProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = BlipForQuestionAnswering.from_pretrained(
            model_name, dtype=dtype, cache_dir=cache_dir
        ).to(device)
        self.model.eval()
        self.device = device
        self.dtype = dtype
        self.normalize = normalize

        self.yes_id = self.processor.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.processor.tokenizer.convert_tokens_to_ids("no")

    @torch.no_grad()
    def _prob_yes(self, images: list[Image.Image], questions: list[str]) -> torch.Tensor:
        """P(first answer token == "yes") for each (image, question) pair.

        `BlipForQuestionAnswering.forward` in this transformers version returns only
        the vision outputs and drops the decoder logits, so we run the three stages
        (vision encoder -> text encoder -> text decoder) directly.

        With `normalize` the returned value is P(yes) / (P(yes) + P(no)),
        renormalised over the two answer tokens; otherwise it is raw P(yes) out of
        the full vocabulary softmax.
        """
        inputs = self.processor(
            images=images, text=questions, return_tensors="pt", padding=True
        ).to(self.device)
        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        image_embeds = self.model.vision_model(pixel_values=pixel_values)[0]
        image_mask = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=self.device)
        question_embeds = self.model.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_mask,
        )[0]

        # One teacher-forced decoder step from BOS.
        bos = self.model.decoder_start_token_id
        dec_ids = torch.full((input_ids.shape[0], 1), bos, device=self.device, dtype=torch.long)
        logits = self.model.text_decoder(
            input_ids=dec_ids,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=attention_mask,
        ).logits

        probs = logits[:, 0, :].float().softmax(dim=-1)
        p_yes = probs[:, self.yes_id]
        if not self.normalize:
            return p_yes
        p_no = probs[:, self.no_id]
        return p_yes / (p_yes + p_no)

    @torch.no_grad()
    def score(self, image: Image.Image, prompt: str) -> float:
        phrases = split_noun_phrases(prompt)
        questions = [f"{p}?" for p in phrases]
        return float(self._prob_yes([image] * len(questions), questions).prod().item())

    @torch.no_grad()
    def score_batch(self, images: list[Image.Image], prompts: list[str]) -> list[float]:
        """Batched scoring: flatten all (image, question) pairs into one forward."""
        flat_images, flat_questions, owner = [], [], []
        for i, (img, prompt) in enumerate(zip(images, prompts)):
            for phrase in split_noun_phrases(prompt):
                flat_images.append(img)
                flat_questions.append(f"{phrase}?")
                owner.append(i)
        if not flat_images:
            return [0.0] * len(images)

        probs = self._prob_yes(flat_images, flat_questions)
        scores = [1.0] * len(images)
        for p, i in zip(probs.tolist(), owner):
            scores[i] *= p
        return scores


class ClipScorer:
    """Cosine similarity between CLIP image and text embeddings."""

    def __init__(self, model_name: str, device="cuda", dtype=torch.float16, cache_dir=None):
        from transformers import CLIPModel, CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = CLIPModel.from_pretrained(model_name, dtype=dtype, cache_dir=cache_dir).to(device)
        self.model.eval()
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def score_batch(self, images: list[Image.Image], prompts: list[str]) -> list[float]:
        inputs = self.processor(
            text=prompts, images=images, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        img = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        txt = self.model.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        # Newer transformers returns an output object here rather than a tensor.
        img = getattr(img, "pooler_output", img)
        txt = getattr(txt, "pooler_output", txt)
        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        return (img.float() * txt.float()).sum(dim=-1).tolist()
