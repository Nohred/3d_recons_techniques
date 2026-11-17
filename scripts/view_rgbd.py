"""Visualiza las 3 vistas RGB + profundidad de un objeto. Uso: python3 scripts/view_rgbd.py <objeto>"""
from pathlib import Path
import hashlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

if len(sys.argv) != 2:
    raise SystemExit(__doc__)
base = ROOT / "data" / sys.argv[1] / "raw" / "rgbd"
if not base.is_dir():
    raise SystemExit(f"No existe {base}")

fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for i in range(3):
    c = base / f"camera_{i:02d}"
    rgb = np.asarray(Image.open(c / "rgb.png"))
    d = np.load(c / "depth.npy")
    valid = np.isfinite(d) & (d > 0)
    print(f"camera_{i:02d}: validos={valid.mean():.2%} "
          f"hash_depth={hashlib.sha256(d.tobytes()).hexdigest()[:12]}")
    axes[0, i].imshow(rgb)
    axes[0, i].set_title(f"camera_{i:02d} RGB")
    im = axes[1, i].imshow(np.where(valid, d, np.nan), cmap="gray")
    axes[1, i].set_title(f"depth {valid.mean():.1%} válido")
    fig.colorbar(im, ax=axes[1, i], fraction=0.046)
    for r in (0, 1):
        axes[r, i].axis("off")
plt.tight_layout()
plt.show()
