"""Dataset over the reward-selected text-image pairs."""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RewardSelectedDataset(Dataset):
    """Text-image pairs with their reward, as produced by scripts/build_dataset.py.

    The reward weights the diffusion loss (GORS, supplementary Eq. 1), so it travels
    with each sample.
    """

    def __init__(
        self,
        roots: list[str],
        resolution: int = 512,
        min_reward: float = 0.0,
        verify_paths: bool = True,
    ):
        """`verify_paths` stats every image on disk, which is slow on a network
        filesystem; pass False for a set that was just built.
        """
        self.records = []
        self.resolution = resolution

        for root in roots:
            img_dir = os.path.join(root, "images")
            for meta in sorted(glob.glob(os.path.join(root, "metadata*.jsonl"))):
                with open(meta) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        if rec.get("reward", 0.0) < min_reward:
                            continue
                        path = os.path.join(img_dir, rec["image"])
                        if verify_paths and not os.path.exists(path):
                            continue
                        self.records.append(
                            {
                                "path": path,
                                "prompt": rec["prompt"],
                                "reward": float(rec.get("reward", 1.0)),
                                "category": rec.get("category", ""),
                            }
                        )
        if not self.records:
            raise RuntimeError(f"no usable records found under {roots}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        img = Image.open(rec["path"]).convert("RGB")
        if img.size != (self.resolution, self.resolution):
            img = img.resize((self.resolution, self.resolution), Image.BICUBIC)
        # [0, 255] HWC -> [-1, 1] CHW, the range the FLUX VAE expects.
        x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1)
        x = x / 127.5 - 1.0
        return {"pixel_values": x, "prompt": rec["prompt"], "reward": rec["reward"]}


def collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "prompt": [b["prompt"] for b in batch],
        "reward": torch.tensor([b["reward"] for b in batch], dtype=torch.float32),
    }
