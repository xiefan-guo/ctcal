"""Check that the BLIP-VQA scorer actually discriminates.

A metric that cannot tell a matching image from a mismatched one would make every
downstream number meaningless, so it is tested on solid colour fields whose content
we control exactly.
"""

import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctcal import config  # noqa: E402
from ctcal.scorer import BlipVQAScorer, split_noun_phrases  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


def solid(color, size=384):
    return Image.new("RGB", (size, size), color)


def half_and_half(left, right, size=384):
    img = Image.new("RGB", (size, size), left)
    for x in range(size // 2, size):
        for y in range(size):
            img.putpixel((x, y), right)
    return img


def main():
    # --- prompt splitting ---------------------------------------------------
    check(
        "splits on 'and'",
        split_noun_phrases("a green bench and a red car") == ["a green bench", "a red car"],
    )
    check("single phrase intact", split_noun_phrases("a red car") == ["a red car"])

    print("\nloading BLIP-VQA...", flush=True)
    scorer = BlipVQAScorer(config.BLIP_VQA, device="cuda", cache_dir=config.hf_hub_cache())

    # --- colour discrimination ---------------------------------------------
    red, blue = solid((220, 30, 30)), solid((30, 30, 220))
    s_red_red = scorer.score(red, "a red background")
    s_red_blue = scorer.score(red, "a blue background")
    print(f"  red image / 'red' = {s_red_red:.4f}   red image / 'blue' = {s_red_blue:.4f}")
    check(
        "matching colour scores higher",
        s_red_red > s_red_blue,
        f"{s_red_red:.4f} > {s_red_blue:.4f}",
    )

    s_blue_blue = scorer.score(blue, "a blue background")
    check("symmetric for blue", s_blue_blue > scorer.score(blue, "a red background"))

    # --- compositional behaviour -------------------------------------------
    # A two-object prompt against an image containing only one object should be
    # penalised, because the score is a product over phrases.
    both = half_and_half((220, 30, 30), (30, 30, 220))
    s_both = scorer.score(both, "a red region and a blue region")
    s_one = scorer.score(red, "a red region and a blue region")
    print(f"  half/half = {s_both:.4f}   red only = {s_one:.4f}")
    check(
        "product penalises a missing object",
        s_both > s_one,
        f"{s_both:.4f} > {s_one:.4f}",
    )

    # --- range and batching -------------------------------------------------
    check("scores in [0,1]", all(0.0 <= s <= 1.0 for s in (s_red_red, s_red_blue, s_both, s_one)))

    batched = scorer.score_batch([red, red], ["a red background", "a blue background"])
    print(f"  batched: {[round(x, 4) for x in batched]}")
    check(
        "batch matches single",
        abs(batched[0] - s_red_red) < 1e-3 and abs(batched[1] - s_red_blue) < 1e-3,
        f"{batched[0]:.4f} vs {s_red_red:.4f}",
    )

    print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURES: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
