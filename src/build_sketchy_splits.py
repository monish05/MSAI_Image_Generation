#!/usr/bin/env python3
"""
Build paired (photo, sketch) rows for Sketchy 256×256 with matching tx folders,
then write stratified train/val/test CSVs keyed by image stem (no sketch leakage).

Default: `data/photo/tx_000000000000` + `data/sketch/tx_000000000000` under the project.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def discover_pairs(
    photo_tx: Path,
    sketch_tx: Path,
) -> list[tuple[str, str, Path, list[Path]]]:
    """
    Returns list of (class_name, stem, photo_path, sketch_paths_sorted).
    """
    rows: list[tuple[str, str, Path, list[Path]]] = []
    if not photo_tx.is_dir():
        raise SystemExit(f"Missing photo dir: {photo_tx}")
    if not sketch_tx.is_dir():
        raise SystemExit(f"Missing sketch dir: {sketch_tx}")

    for class_dir in sorted(photo_tx.iterdir()):
        if not class_dir.is_dir():
            continue
        cls = class_dir.name
        sk_cls = sketch_tx / cls
        if not sk_cls.is_dir():
            print(f"warn: no sketch folder for class {cls}")
            continue
        for photo in sorted(class_dir.glob("*.jpg")):
            stem = photo.stem
            sketches = sorted(sk_cls.glob(f"{stem}-*.png"))
            if not sketches:
                print(f"warn: no sketches for {cls}/{stem}")
                continue
            rows.append((cls, stem, photo, sketches))
    return rows


def simpler_stratified_split(
    pairs: list[tuple[str, str, Path, list[Path]]],
    rng: random.Random,
    train_frac: float,
    val_frac: float,
) -> dict[tuple[str, str], str]:
    """Per-class shuffle of stems, then contiguous slices for train/val/test."""
    by_class: dict[str, list[tuple[str, Path, list[Path]]]] = defaultdict(list)
    for cls, stem, photo, sketches in pairs:
        by_class[cls].append((stem, photo, sketches))

    out: dict[tuple[str, str], str] = {}
    for cls, items in sorted(by_class.items()):
        rng.shuffle(items)
        n = len(items)
        if n == 1:
            n_train, n_val, n_test = 1, 0, 0
        elif n == 2:
            n_train, n_val, n_test = 1, 1, 0
        else:
            # Integer fractions; remainder goes to test
            n_train = max(1, int(n * train_frac))
            n_val = max(1, int(n * val_frac))
            n_test = n - n_train - n_val
            if n_test < 1:
                n_test = 1
                need = n_train + n_val + n_test - n
                while need > 0 and n_train > 1:
                    n_train -= 1
                    need -= 1
                while need > 0 and n_val > 1:
                    n_val -= 1
                    need -= 1
                while need > 0:
                    n_train = max(1, n_train - 1)
                    need -= 1
            assert n_train + n_val + n_test == n, (cls, n, n_train, n_val, n_test)

        for i, (stem, _, _) in enumerate(items):
            if i < n_train:
                out[(cls, stem)] = "train"
            elif i < n_train + n_val:
                out[(cls, stem)] = "val"
            else:
                out[(cls, stem)] = "test"
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
        help="Contains photo/ and sketch/ subdirs (each with tx folders).",
    )
    p.add_argument(
        "--tx",
        type=str,
        default="tx_000000000000",
        help="Use same tx under photo/ and sketch/.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "metadata" / "sketchy_tx000",
    )
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    test_frac = 1.0 - args.train_frac - args.val_frac
    if test_frac < -1e-6 or abs(test_frac - round(test_frac, 6)) > 1e-3:
        raise SystemExit(
            f"train_frac + val_frac must be <= 1; test = {test_frac:.4f}"
        )

    photo_tx = args.data_root / "photo" / args.tx
    sketch_tx = args.data_root / "sketch" / args.tx
    pairs = discover_pairs(photo_tx, sketch_tx)
    if not pairs:
        raise SystemExit("No pairs found; check --data-root and --tx.")

    rng = random.Random(args.seed)
    split_map = simpler_stratified_split(
        pairs, rng, args.train_frac, args.val_frac
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    project_root = args.out_dir.parents[1]

    def rel(p: Path) -> str:
        return str(p.resolve().relative_to(project_root.resolve()))

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []

    stem_counts = {"train": 0, "val": 0, "test": 0}
    pair_counts = {"train": 0, "val": 0, "test": 0}

    for cls, stem, photo, sketches in pairs:
        sp = split_map[(cls, stem)]
        stem_counts[sp] += 1
        for sk in sketches:
            row = {
                "split": sp,
                "class": cls,
                "stem": stem,
                "photo_relpath": rel(photo),
                "sketch_relpath": rel(sk),
            }
            pair_counts[sp] += 1
            if sp == "train":
                train_rows.append(row)
            elif sp == "val":
                val_rows.append(row)
            else:
                test_rows.append(row)

    fieldnames = ["split", "class", "stem", "photo_relpath", "sketch_relpath"]

    def write_csv(name: str, rows: list[dict[str, str]]) -> Path:
        path = args.out_dir / name
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        return path

    write_csv("train.csv", train_rows)
    write_csv("val.csv", val_rows)
    write_csv("test.csv", test_rows)

    stats = {
        "tx": args.tx,
        "data_root_relpath": rel(args.data_root),
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": round(1.0 - args.train_frac - args.val_frac, 4),
        "num_classes": len({c for c, _, _, _ in pairs}),
        "unique_photos": len(pairs),
        "stems_per_split": stem_counts,
        "sketch_photo_pairs_per_split": pair_counts,
    }
    stats_path = args.out_dir / "split_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"Wrote {args.out_dir}/train.csv, val.csv, test.csv")
    print(f"Wrote {stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
