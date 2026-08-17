#!/usr/bin/env python3
"""Build a symlinked decoder panel and run its fixed-sampler analysis."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path(
    "/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/"
    "local-center-decoder-followup-8x100k/"
    "1ecf2c66b659d0f0b335405e231dda5e4e7a40ae/v1"
)
ORIGINAL_PANEL = RUN_ROOT / "samples" / "step100000"
RECOVERY_SAMPLE = RUN_ROOT / "samples" / "step100000-recovery-v4" / "coarse12_residue8_331283"
PANEL_ROOT = RUN_ROOT / "samples" / "step100000-recovery-v4-panel"
OUTPUT = RUN_ROOT / "analysis" / "milestone_step100000-recovery-v4.json"
VARIANTS = (
    "atom_decoder_double_33836", "coarse12_decoder_double_331266",
    "coarse12_residue6_331263", "coarse12_residue8_331283",
    "coarse12_residue8_decoder_double_331286", "decoder_double_33866",
    "residue_decoder_double_33863", "residue_deeper_33883",
)


def _link_variant(variant: str) -> None:
    source = RECOVERY_SAMPLE if variant == RECOVERY_SAMPLE.name else ORIGINAL_PANEL / variant
    target = PANEL_ROOT / variant
    if not source.is_dir() or not (source / "manifest.json").is_file():
        raise FileNotFoundError(f"validated sample directory is missing: {source}")
    if target.exists() or target.is_symlink():
        if not target.is_symlink() or Path(os.readlink(target)) != source:
            raise FileExistsError(f"recovery panel target is not the expected symlink: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=True)


def main() -> None:
    for variant in VARIANTS:
        _link_variant(variant)
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "analyze_sequence_diversity_16x100k.py"),
            "--sample-root", str(PANEL_ROOT), "--step", "100000",
            "--output", str(OUTPUT), "--decoder-followup",
            "--control", "decoder_double_33866",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
