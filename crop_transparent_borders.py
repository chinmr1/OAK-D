#!/usr/bin/env python3
"""
Batch-crop PNGs by removing fully/mostly transparent borders.

What it does:
- For each PNG in --in, finds the tight bounding box of pixels whose alpha >= --min-alpha.
- Crops to that box (optionally with --pad pixels of padding).
- Saves the cropped PNG to the same relative path under --out.

Notes:
- Uses the alpha channel only, so it works even if RGB values inside transparent pixels are non-zero.
- If an image is fully transparent (no pixels above threshold), it copies the original to output.
"""



# [I M P O R T A N T]
# Usage example:
# python3 .\crop_transparent_borders.py --in Alpha_Cube --out Alpha_Test --pad 0 --min-alpha 1





from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image


def compute_alpha_bbox(img: Image.Image, min_alpha: int) -> tuple[int, int, int, int] | None:
    """Return bbox (l, u, r, d) where alpha >= min_alpha. None if nothing matches."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # Alpha channel as L-mode image
    alpha = img.getchannel("A")

    # Threshold alpha so faint near-transparent noise can be excluded if desired.
    # If min_alpha == 1, this includes all non-zero alpha.
    if min_alpha <= 1:
        return alpha.getbbox()

    mask = alpha.point(lambda a: 255 if a >= min_alpha else 0)
    return mask.getbbox()


def clamp_bbox(bbox: tuple[int, int, int, int], w: int, h: int, pad: int) -> tuple[int, int, int, int]:
    l, u, r, d = bbox
    l = max(0, l - pad)
    u = max(0, u - pad)
    r = min(w, r + pad)
    d = min(h, d + pad)
    return (l, u, r, d)


def iter_pngs(root: Path, recursive: bool):
    if recursive:
        yield from root.rglob("*.png")
        yield from root.rglob("*.PNG")
    else:
        yield from root.glob("*.png")
        yield from root.glob("*.PNG")


def main() -> int:
    p = argparse.ArgumentParser(description="Crop excess transparent borders from PNGs (batch).")
    p.add_argument("--in", dest="in_dir", required=True, help="Input directory containing PNGs")
    p.add_argument("--out", dest="out_dir", required=True, help="Output directory to write cropped PNGs")
    p.add_argument("--recursive", action="store_true", help="Process PNGs in subfolders too")
    p.add_argument("--pad", type=int, default=0, help="Padding (pixels) to keep around the cropped object")
    p.add_argument("--min-alpha", type=int, default=1,
                   help="Alpha threshold (1-255). Pixels with alpha >= threshold are kept. Default 1.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files in output")
    args = p.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"Input directory not found or not a directory: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    cropped = 0
    copied = 0
    skipped = 0

    for src in iter_pngs(in_dir, args.recursive):
        total += 1
        rel = src.relative_to(in_dir)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not args.overwrite:
            print(f"[skip exists] {rel}")
            skipped += 1
            continue

        try:
            with Image.open(src) as img:
                img = img.convert("RGBA")  # ensure alpha channel exists
                bbox = compute_alpha_bbox(img, args.min_alpha)
                if bbox is None:
                    # Fully transparent (or nothing above threshold): just copy original
                    shutil.copy2(src, dst)
                    print(f"[copy (empty)] {rel}")
                    copied += 1
                    continue

                bbox = clamp_bbox(bbox, img.width, img.height, args.pad)

                # If bbox equals full image, no crop needed; still save to output for consistency.
                out_img = img.crop(bbox)
                out_img.save(dst, format="PNG")
                print(f"[crop] {rel}  ->  {bbox}")
                cropped += 1

        except Exception as e:
            print(f"[error] {rel}: {e}")
            # Keep going on errors
            continue

    print(f"\nDone. total={total}, cropped={cropped}, copied={copied}, skipped={skipped}, out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
