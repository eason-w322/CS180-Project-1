import os
from pathlib import Path
import time
import numpy as np
from skimage import io, img_as_float, img_as_ubyte
from utils import split_bgr, align_single, align_pyramid, apply_shift, crop_border

ROOT = Path(__file__).resolve().parent.parent
from contextlib import contextmanager

@contextmanager
def section(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[TIMER] {name}: {dt:.3f} s")


def run_one(im_path: Path, out_path: Path, win: int = 15, crop_frac: float = 0.15):
    if not im_path.exists():
        raise FileNotFoundError(f"Input image not found: {im_path}")
    t_total = time.perf_counter() 
    im = io.imread(str(im_path))
    im = img_as_float(im)
    B, G, R = split_bgr(im)
    print("Slice shape:", B.shape)
    use_pyramid = max(B.shape) > 800  
    if use_pyramid:
        print("Aligning with PYRAMID (gradients)…")
        dG = align_pyramid(B, G, levels=None, base_win=12, refine_win=3,
                           crop_frac=0.15, scale=2, use_grad=False)
        dR = align_pyramid(B, R, levels=None, base_win=12, refine_win=3,
                           crop_frac=0.15, scale=2, use_grad=False)
    else:
        print("Aligning with SINGLE-SCALE (gradients)…")
        dG = align_single(B, G, win=win, crop_frac=crop_frac, use_grad=False)
        dR = align_single(B, R, win=win, crop_frac=crop_frac, use_grad=False)

    print(f"Offsets  G(dy,dx)={dG}  R(dy,dx)={dR}")

    # 4) apply shifts
    G_shift = apply_shift(G, dG)   
    R_shift = apply_shift(R, dR)

    # 5) stack RGB and save
    rgb = np.stack([R_shift, G_shift, B], axis=2)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)

    # small final crop to clean scan edges
    rgb = crop_border(rgb, frac=0.06)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(str(out_path), img_as_ubyte(np.clip(rgb, 0.0, 1.0)))
    print("Saved:", out_path)

    print(f"[TIMER] TOTAL: {time.perf_counter() - t_total:.3f} s")


if __name__ == "__main__":
    in_file  = ROOT / "data" / "emir.tif" #input file here replace image_name 
    out_file = ROOT / "results" / "emir_rgb.jpg" #output file here replace image name
    run_one(in_file, out_file, win=15, crop_frac=0.15)
