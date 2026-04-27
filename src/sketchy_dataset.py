"""Load (sketch, photo) pairs from metadata CSV manifests."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import resize


def _to_float01(img: torch.Tensor) -> torch.Tensor:
    return img.float() / 255.0


def sketch_to_gray(sk: torch.Tensor) -> torch.Tensor:
    """[C,H,W] float [0,1] -> [1,H,W] luminance."""
    if sk.shape[0] >= 3:
        r, g, b = sk[0], sk[1], sk[2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        return y.unsqueeze(0)
    return sk[:1]


def normalize_m11(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


class SketchyPairDataset(Dataset):
    """One CSV row = one (photo, sketch) training pair."""

    def __init__(
        self,
        csv_path: Path,
        root: Path,
        image_size: int | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root).resolve()
        self.image_size = image_size
        self.rows: list[dict[str, str]] = []
        with Path(csv_path).open(newline="") as f:
            for row in csv.DictReader(f):
                self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        photo_path = self.root / row["photo_relpath"]
        sketch_path = self.root / row["sketch_relpath"]
        photo = read_image(str(photo_path))
        if photo.shape[0] == 1:
            photo = photo.repeat(3, 1, 1)
        elif photo.shape[0] == 4:
            photo = photo[:3]
        sketch = read_image(str(sketch_path))

        photo_f = _to_float01(photo)
        sketch_f = _to_float01(sketch)
        sketch_gray = sketch_to_gray(sketch_f)

        if self.image_size is not None and (
            photo_f.shape[-2] != self.image_size
            or photo_f.shape[-1] != self.image_size
        ):
            photo_f = resize(photo_f, [self.image_size, self.image_size])
            sketch_gray = resize(sketch_gray, [self.image_size, self.image_size])

        return {
            "photo": normalize_m11(photo_f),
            "sketch": normalize_m11(sketch_gray),
            "class": row["class"],
            "stem": row["stem"],
        }
