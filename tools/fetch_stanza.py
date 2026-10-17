"""Fetch the Stanza English models CTCal needs into a local resources dir.

Stanza's own downloader pulls from a fixed endpoint, so this script pulls the same
files through huggingface_hub instead (which honours HF_ENDPOINT and any mirror)
and lays them out the way `stanza.Pipeline(download_method=None)` expects.

Run this before training, then point STANZA_RESOURCES_DIR at the output directory
(or leave it unset and stage into `$HF_HOME/stanza_resources`, which
`ctcal.config.stanza_dir()` resolves to).
"""

import json
import os
import shutil

from huggingface_hub import hf_hub_download

ROOT = os.environ.get("STANZA_RESOURCES_DIR") or os.path.join(
    os.environ.get("CTCAL_HF_HOME")
    or os.environ.get("HF_HOME")
    or os.path.expanduser("~/.cache/huggingface"),
    "stanza_resources",
)
REPO = "stanfordnlp/stanza-en"

# combined_nocharlm avoids pulling the (large) character language models, but the
# pos/depparse taggers still need the conll17 word vectors they were trained against.
FILES = {
    "tokenize/combined.pt": "tokenize",
    "pos/combined_nocharlm.pt": "pos",
    "lemma/combined_nocharlm.pt": "lemma",
    "depparse/combined_nocharlm.pt": "depparse",
    "pretrain/conll17.pt": "pretrain",
}


def main():
    for rel in FILES:
        src = hf_hub_download(REPO, f"models/{rel}")
        dst = os.path.join(ROOT, "en", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy(src, dst)
        print(f"OK {dst} {os.path.getsize(dst) // 1024} KB", flush=True)

    # Minimal resources.json so stanza resolves the package without a network call.
    # The "packages" section is required by stanza >= 1.12's processor resolution.
    default_processors = {
        "tokenize": "combined",
        "pos": "combined_nocharlm",
        "lemma": "combined_nocharlm",
        "depparse": "combined_nocharlm",
    }
    # pos/depparse load their word vectors via a per-package "dependencies" entry.
    pretrain_dep = {"dependencies": [{"model": "pretrain", "package": "conll17"}]}
    res = {
        "en": {
            "lang_name": "English",
            "tokenize": {"combined": {}},
            "pos": {"combined_nocharlm": dict(pretrain_dep)},
            "lemma": {"combined_nocharlm": {}},
            "depparse": {"combined_nocharlm": dict(pretrain_dep)},
            "pretrain": {"conll17": {}},
            "packages": {"default": dict(default_processors)},
            "default_processors": default_processors,
            "default_dependencies": {},
            "default_md5": "",
        },
        "url": "local",
    }
    with open(os.path.join(ROOT, "resources.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("wrote resources.json")


if __name__ == "__main__":
    main()
