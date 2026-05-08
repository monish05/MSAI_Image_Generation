from __future__ import annotations

import csv
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/szb9536/genai_test_4")
DATA_ROOT = PROJECT_ROOT / "data"
PYTHON_EXE = sys.executable


def _has_jpgs(root: Path) -> bool:
    return bool(list(root.glob("*.jpg"))) or bool(list((root / "img_align_celeba").glob("*.jpg")))


def _pick_image_root(data_root: Path) -> Path:
    candidates = [
        data_root / "img_align_celeba" / "img_align_celeba",
    ]
    for cand in candidates:
        if cand.is_dir() and _has_jpgs(cand):
            return cand
    for cand in candidates:
        if cand.is_dir():
            return cand
    return candidates[1]


IMAGE_ROOT = _pick_image_root(DATA_ROOT)

BASE_ARGS: dict[str, str | None] = {
    "data-root": str(DATA_ROOT),
    "image-root": str(IMAGE_ROOT),
    "image-size": "64",
    "batch-size": "32",
    "epochs": "100",
    "lr": "2e-4",
    "workers": "4",
    "seed": "42",
    "timesteps": "1000",
    "beta-schedule": "cosine",
    "min-snr-gamma": "0",
    "drop-sketch-prob": "0.1",
    "guidance-scale": "1.5",
    "sample-steps": "200",
    "sample-every": "2000",
    "lpips-start-frac": "0.1",
    "color-loss-weight": "0.02",
    "color-loss-start-frac": "0.6",
    "color-loss-ramp-steps": "5000",
    "early-stop-patience": "0",
    "early-stop-min-delta": "0",
    "max-train-images": "70000",
    "amp": None,
}


def build_cmd(run: dict) -> tuple[list[str], Path]:
    args = dict(BASE_ARGS)
    args.update(run.get("overrides", {}))
    save_dir = PROJECT_ROOT / "checkpoints" / f"hp_{run['name']}"
    args["save-dir"] = str(save_dir)
    cmd: list[str] = [PYTHON_EXE, "-m", "src.train"]
    for k, v in args.items():
        if v is None:
            cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])
    cmd.extend(run.get("extra_flags", []))
    return cmd, save_dir


def stream_run(cmd: list[str], cwd: Path) -> int:
    print("=== command ===")
    print(shlex.join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    return proc.wait()


def _last_val_row(metrics_csv: Path) -> dict[str, str]:
    if not metrics_csv.is_file():
        return {}
    rows = list(csv.DictReader(metrics_csv.open(newline="")))
    for r in reversed(rows):
        if (r.get("val_loss") or "").strip():
            return r
    return {}


def run_case(case_run: dict, run_now: bool = True) -> None:
    cmd, save_dir = build_cmd(case_run)
    print("project=", PROJECT_ROOT)
    print("python=", PYTHON_EXE)
    print("data_root=", DATA_ROOT)
    print("image_root=", IMAGE_ROOT)
    print("run=", case_run["name"])
    print("save_dir=", save_dir)
    print("cmd=", shlex.join(cmd))

    if run_now:
        rc = stream_run(cmd, PROJECT_ROOT)
        print("returncode=", rc)

    row = _last_val_row(save_dir / "metrics.csv")
    print("last_step=", row.get("step"))
    print("val_loss=", row.get("val_loss"))
    print("val_psnr=", row.get("val_psnr"))
