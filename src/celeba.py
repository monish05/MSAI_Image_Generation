"""CelebA aligned faces under data/: partition from list_eval_partition.csv."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .sketch import photo_bgr_uint8_to_sketch_gray, sketch_gray_uint8_to_tensor01

PART_TRAIN = 0
PART_VAL = 1
PART_TEST = 2


def resolve_image_root(data_root: Path) -> Path:
    dire = data_root / "img_align_celeba"
    if not dire.is_dir():
        raise FileNotFoundError(f"Missing {dire}")
    if list(dire.glob("*.jpg")):
        return dire
    inner = dire / "img_align_celeba"
    if inner.is_dir() and list(inner.glob("*.jpg")):
        return inner
    raise FileNotFoundError(f"No .jpg under {dire}")


def list_image_ids_sorted(data_root: Path, partition: int, image_root: Path | None = None) -> list[str]:
    root = Path(data_root).resolve()
    ir = Path(image_root).resolve() if image_root else resolve_image_root(root)
    part_csv = root / "list_eval_partition.csv"
    if not part_csv.is_file():
        raise FileNotFoundError(part_csv)
    rows: list[str] = []
    with part_csv.open(newline="") as f:
        for r in csv.DictReader(f):
            if int(r["partition"]) != partition:
                continue
            fname = r["image_id"].strip()
            if (ir / fname).is_file():
                rows.append(fname)
    rows.sort()
    return rows


class CelebSketchDataset(Dataset):
    """Paired photo + synthetic sketch per aligned face."""

    def __init__(
        self,
        data_root: Path,
        partition: int,
        image_size: int,
        *,
        blur_ksize: int = 21,
        max_images: int | None = None,
        image_root: Path | None = None,
        only_filenames: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root).resolve()
        self.image_root = (
            Path(image_root).resolve() if image_root else resolve_image_root(self.data_root)
        )
        rows = list_image_ids_sorted(self.data_root, partition, self.image_root)
        if only_filenames is not None:
            want = frozenset(only_filenames)
            rows = [f for f in rows if f in want]
            rows.sort()
        if max_images is not None:
            rows = rows[: max_images]
        self._files = rows
        self.image_size = image_size
        self.blur_ksize = blur_ksize

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.image_root / self._files[idx]
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise RuntimeError(f"Unreadable image {path}")

        sz = self.image_size
        bgr = cv2.resize(bgr, (sz, sz), interpolation=cv2.INTER_AREA)
        sk_uint8 = photo_bgr_uint8_to_sketch_gray(bgr, blur_ksize=self.blur_ksize)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        photo = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        sketch = sketch_gray_uint8_to_tensor01(sk_uint8)
        photo_m11 = photo * 2.0 - 1.0
        sketch_m11 = sketch * 2.0 - 1.0
        return {"photo": photo_m11, "sketch": sketch_m11}


def sample_fixed_manifest(
    data_root: Path,
    partition: int,
    *,
    image_root: Path | None,
    max_count: int,
    seed: int,
    out_json: Path | None = None,
) -> list[str]:
    names = list_image_ids_sorted(data_root, partition, image_root)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(names), generator=g).tolist()
    sel = sorted(names[i] for i in idx[: max_count])
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"image_ids": sel, "seed": seed, "partition": partition}
        out_json.write_text(json.dumps(payload, indent=2))
    return sel
