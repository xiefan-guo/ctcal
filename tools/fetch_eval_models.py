"""Pre-fetch the scoring models into the Hugging Face cache.

BLIP-VQA is T2I-CompBench++'s metric for the attribute-binding categories
(color / shape / texture) and is also the reward used for GORS-style dataset
construction. CLIP is used for the auxiliary CLIPScore.

Run this once before `build_dataset.py` / `evaluate.py` if you would rather
download up front than on first use. Honours HF_ENDPOINT, so a mirror works.
"""

import os

from huggingface_hub import snapshot_download

MODELS = [
    ("Salesforce/blip-vqa-capfilt-large", None),
    ("openai/clip-vit-large-patch14", None),
]


def main():
    for repo, revision in MODELS:
        path = snapshot_download(
            repo,
            revision=revision,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "*.bin"],
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        size = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk(path)
            for f in fs
        )
        print(f"OK {repo} -> {path} ({size / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
