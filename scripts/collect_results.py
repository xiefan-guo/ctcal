"""Gather every results/*/summary.json into one comparison table."""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctcal import config  # noqa: E402


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(config.REPO_ROOT, "results", "*", "summary.json"))):
        with open(path) as f:
            rows.append(json.load(f))
    if not rows:
        print("no results found")
        return

    categories = sorted({r["category"] for r in rows})
    # Reading order: pre-trained, baseline fine-tune, then ours.
    order = ["baseline", "gors", "ctcal"]
    tags = sorted({r["tag"] for r in rows})
    tags.sort(key=lambda t: (order.index(t) if t in order else len(order), t))

    for metric in ("blip_vqa", "clip"):
        print(f"\n=== {metric} ===")
        header = "method".ljust(22) + "".join(c.ljust(12) for c in categories)
        print(header)
        print("-" * len(header))
        base = {}
        for tag in tags:
            line = tag.ljust(22)
            for cat in categories:
                hit = [r for r in rows if r["tag"] == tag and r["category"] == cat]
                if not hit:
                    line += "-".ljust(12)
                    continue
                v = hit[0][metric]
                if tag == "baseline":
                    base[cat] = v
                    line += f"{v:.4f}".ljust(12)
                else:
                    ref = base.get(cat)
                    delta = f" ({(v - ref) / ref * 100:+.1f}%)" if ref else ""
                    line += (f"{v:.4f}" + delta).ljust(12)
            print(line)

    out = os.path.join(config.REPO_ROOT, "results", "all_results.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
