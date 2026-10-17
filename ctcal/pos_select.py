"""Part-of-speech-based cross-attention map selection (paper Sec. 3.2).

Only tokens that carry spatial semantics are used for calibration; nouns are the
primary target.

This module also maps a *word* to the T5 sub-token positions that spell it. FLUX's
text stream is T5/SentencePiece, so a word can span several tokens.
"""

from __future__ import annotations

import functools
import re

# Directional / relational nouns that Stanza tags as NOUN but which carry no object
# semantics; a blacklist is the mitigation the paper describes for this limitation.
NON_SPATIAL_NOUNS = {
    "top", "bottom", "left", "right", "side", "front", "back", "middle",
    "center", "centre", "edge", "corner", "row", "line", "pair", "group",
    "number", "kind", "sort", "type", "piece", "part", "lot", "bunch",
}


@functools.lru_cache(maxsize=1)
def get_nlp(stanza_dir: str | None = None, use_gpu: bool = False):
    """Build the Stanza pipeline (cached; the models are ~200MB).

    `stanza_dir` is where `tools/fetch_stanza.py` staged the models; leave it None
    to let Stanza resolve its own resources directory.
    """
    import stanza

    # The lemma model shipped for this stanza build is version-incompatible, and
    # CTCal only needs `pos`/`deprel`, so we use the identity lemmatiser (which
    # depparse requires as a prerequisite but whose output we never read).
    return stanza.Pipeline(
        "en",
        dir=stanza_dir,
        download_method=None,
        use_gpu=use_gpu,
        logging_level="ERROR",
        processors={
            "tokenize": "combined",
            "pos": "combined_nocharlm",
            "lemma": "identity",
            "depparse": "combined_nocharlm",
        },
    )


def select_words(prompt: str, nlp, filter_non_spatial: bool = True) -> list[str]:
    """The words whose attention maps CTCal should align.

    NOUN tokens, skipping `compound` dependents so that a multi-word entity
    contributes one map rather than one per modifier.
    """
    picked = []
    for sent in nlp(prompt).sentences:
        for word in sent.words:
            if word.pos != "NOUN" or word.deprel == "compound":
                continue
            if filter_non_spatial and word.text.lower() in NON_SPATIAL_NOUNS:
                continue
            picked.append(word.text)
    return picked


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def word_to_token_indices(
    prompt: str, words: list[str], tokenizer, max_length: int
) -> list[list[int]]:
    """Map each selected word to its T5 token positions in the encoded prompt.

    A word maps to a *set* of positions. We reconstruct each token's surface form
    and scan left to right, matching greedily so repeated words land on distinct
    occurrences.
    """
    enc = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors=None,
    )
    pieces = tokenizer.convert_ids_to_tokens(enc["input_ids"])

    # SentencePiece marks a word boundary with U+2581; strip it for matching and
    # keep track of which token index each surface character came from.
    surface, owner = "", []
    for pos, piece in enumerate(pieces):
        if piece in tokenizer.all_special_tokens:
            continue
        cleaned = _norm(piece.replace("▁", ""))
        surface += cleaned
        owner.extend([pos] * len(cleaned))

    result, cursor = [], 0
    for word in words:
        target = _norm(word)
        if not target:
            result.append([])
            continue
        at = surface.find(target, cursor)
        if at < 0:
            at = surface.find(target)  # fall back to the first occurrence
        if at < 0:
            result.append([])
            continue
        result.append(sorted(set(owner[at : at + len(target)])))
        cursor = at + len(target)
    return result
