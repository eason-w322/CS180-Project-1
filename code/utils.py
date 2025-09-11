import numpy as np
from skimage.transform import resize

def split_bgr(glass):
    h = glass.shape[0] // 3
    B = glass[0:h, :]
    G = glass[h:2*h, :]
    R = glass[2*h:3*h, :]
    return B, G, R

def crop_center(img, frac):
    h, w = img.shape[:2]
    top = int(frac * h)
    bottom = int((1.0-frac) * h)
    left = int(frac * w)
    right = int((1.0-frac) * w)
    return img[top:bottom,left:right]

def score_ncc(a, b) -> float:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    # dot(a, b) / (||a|| * ||b||)
    num   = float((a * b).sum())
    den   = ( (a*a).sum()**0.5 * (b*b).sum()**0.5 ) + 1e-8
    return num / den

def align_single(img1, img2, win, crop_frac=0.15, use_grad=False):
    r = crop_center(img1, crop_frac)
    m = crop_center(img2, crop_frac)
    if use_grad:
        r = grad_mag(r)
        m = grad_mag(m)

    best_score = -1e18
    best_align = (0, 0)
    for dy in range(-win, win + 1):
        for dx in range(-win, win + 1):
            m_shift = np.roll(m, (dy, dx), axis=(0, 1))
            s = score_ncc(r, m_shift)
            if s > best_score:
                best_score, best_align = s, (dy, dx)
    return best_align

def apply_shift(img, shift):  
    dy, dx = shift
    out = np.roll(img, (dy, dx), axis=(0, 1))
    return out

def downsample(img, scale):
    h,w = img.shape[:2]
    new_h = max(1, h//scale)
    new_w = max(1, w//scale)
    out = resize(img, (new_h, new_w), order = 1, anti_aliasing = True, preserve_range = True)
    return out.astype(np.float32)

def pick_levels(img_shape, min_size, max_levels, scale):
    h, w = img_shape
    levels = 1
    while levels < max_levels and min(h,w) >= min_size * (scale**levels):
        levels += 1
    return levels

def align_pyramid(ref, mov,levels, base_win = 12, refine_win = 3, crop_frac = 0.15, scale = 2, use_grad: bool = False):
    refs = [ref]
    movs = [mov]
    if levels is None:
        levels = pick_levels(ref.shape, min_size=200, max_levels=6, scale=scale)
    for _ in range(1, levels):
        refs.append(downsample(refs[-1], scale=scale))
        movs.append(downsample(movs[-1], scale=scale))
    refs = refs[::-1]
    movs = movs[::-1]
    dy, dx = align_single(refs[0], movs[0], win=base_win, crop_frac=crop_frac, use_grad=use_grad)
    for lvl in range(1, len(refs)):
        dy *= scale
        dx *= scale
        r = crop_center(refs[lvl], crop_frac)
        m = crop_center(movs[lvl], crop_frac)
        if use_grad:
            r = grad_mag(r)
            m = grad_mag(m)
        best_s = -1e18
        best = (dy, dx)
        for ddy in range(-refine_win, refine_win + 1):
            for ddx in range(-refine_win, refine_win + 1):
                trial = (dy + ddy, dx + ddx)
                s = score_ncc(r, np.roll(m, trial, axis=(0, 1)))
                if s > best_s:
                    best_s, best = s, trial
        dy, dx = best
    return int(dy), int(dx)


def crop_border(img, frac: float = 0.02):
    h, w = img.shape[:2]
    py = int(frac * h)
    px = int(frac * w)
    return img[py:h - py, px:w - px]

def grad_mag(img):
    x = img.astype(np.float32, copy=False)
    dx = x[:, 1:] - x[:, :-1]
    dy = x[1:, :] - x[:-1, :]
    dx = np.pad(dx, ((0, 0), (0, 1)), mode="edge")
    dy = np.pad(dy, ((0, 1), (0, 0)), mode="edge")
    g = np.hypot(dx, dy)  # sqrt(dx^2 + dy^2)
    g = g - g.mean()
    g = g / (g.std() + 1e-8)
    return g
