"""Check noun selection and word -> T5 token alignment.

Getting the token indices wrong would silently align the wrong attention maps, so
this verifies the mapping against the FLUX T5 tokenizer directly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctcal import config  # noqa: E402
from ctcal.pos_select import (  # noqa: E402
    get_nlp,
    select_words,
    word_to_token_indices,
)

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


def main():
    nlp = get_nlp(config.stanza_dir())

    # --- noun selection -----------------------------------------------------
    cases = [
        ("a red car and a yellow clock", ["car", "clock"]),
        ("a green backpack and a blue banana", ["backpack", "banana"]),
        ("a wooden door and a glass jar", ["door", "jar"]),
    ]
    for prompt, expected in cases:
        got = select_words(prompt, nlp)
        check(f"nouns of {prompt!r}", got == expected, f"got {got}")

    # Directional nouns must be filtered (paper Appendix E).
    spatial = select_words("a horse on the left of a car", nlp)
    check(
        "filters directional noun 'left'",
        "left" not in spatial and "horse" in spatial and "car" in spatial,
        f"got {spatial}",
    )
    # --- token alignment ----------------------------------------------------
    # Use the same loader FluxPipeline uses, so the vocab matches training exactly.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.flux_path(), subfolder="tokenizer_2")
    maxlen = config.T5_MAX_LENGTH
    probe = tok.convert_ids_to_tokens(tok("a red car")["input_ids"])
    check("tokenizer vocab loaded", "<unk>" not in probe, f"pieces={probe}")

    prompt = "a red car and a yellow clock"
    words = select_words(prompt, nlp)
    idxs = word_to_token_indices(prompt, words, tok, maxlen)
    print(f"  {prompt!r} -> words={words} token_idxs={idxs}")
    check("every word maps somewhere", all(len(i) > 0 for i in idxs), str(idxs))
    check("distinct words map to distinct tokens", idxs[0] != idxs[1])
    check("indices ascend with word order", max(idxs[0]) < min(idxs[1]))

    # The mapped tokens must actually spell the word.
    pieces = tok.convert_ids_to_tokens(
        tok(prompt, padding="max_length", max_length=maxlen, truncation=True)["input_ids"]
    )
    for word, ids in zip(words, idxs):
        spelled = "".join(pieces[i].replace("▁", "") for i in ids).lower()
        check(f"tokens for {word!r} spell it", word.lower() in spelled, f"got {spelled!r}")

    # A multi-token word should map to several positions.
    long_prompt = "a metallic desk lamp and a leather jacket"
    lw = select_words(long_prompt, nlp)
    li = word_to_token_indices(long_prompt, lw, tok, maxlen)
    print(f"  {long_prompt!r} -> words={lw} token_idxs={li}")
    check("multi-word prompt maps all nouns", all(len(i) > 0 for i in li), str(li))

    # Repeated words must land on distinct occurrences.
    rep = "a cat and a cat"
    ri = word_to_token_indices(rep, ["cat", "cat"], tok, maxlen)
    check("repeated word maps to distinct positions", ri[0] != ri[1], str(ri))

    # Indices must stay inside the encoded sequence.
    check("indices within bounds", all(i < maxlen for ids in idxs for i in ids))

    print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURES: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
