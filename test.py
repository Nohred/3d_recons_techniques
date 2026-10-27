from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

OBJETO = "sneaker"
base = Path.home() / "Escritorio/3D_reconstruction/data" / OBJETO / "raw/rgbd"

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
    # Mostrar la profundidad en escala de grises, como en RealSense.
    depth = np.where(valid, d, np.nan)
    im = axes[1, i].imshow(depth, cmap="gray")
    axes[1, i].set_title(f"depth {valid.mean():.1%} válido")
    fig.colorbar(im, ax=axes[1, i], fraction=0.046)
    for r in (0, 1):
        axes[r, i].axis("off")
plt.tight_layout()
plt.show()