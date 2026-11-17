"""Tabla de avance por objeto: qué salidas de cada etapa existen en data/<objeto>/.

Uso: python3 scripts/status.py [objeto ...] [--summary]
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.methods import METHODS, clean_path, raw_path  # noqa: E402

REPORTS = ROOT / "reports" / "evaluation"


def checks(d: Path) -> dict[str, bool]:
    rgbd = d / "raw" / "rgbd"
    rec = d / "reconstruction"
    c = {
        "scene": (d / "scene" / "scene.usd").is_file(),
        "orbit": (d / "raw" / "orbit.mp4").is_file(),
        "rgbd": all((rgbd / f"camera_{i:02d}" / "depth.npy").is_file() for i in range(3)),
        "calib": (d / "raw" / "calibration" / "calibration_inputs.json").is_file(),
        "gt": (d / "ground_truth" / "ground_truth.ply").is_file(),
        "colmap": (rec / ".process.done").is_file(),
    }
    for m in METHODS:
        r, k = raw_path(d, m), clean_path(d, m)
        c[m] = "E" if (REPORTS / f"{d.name}__{m}.json").is_file() else (
            "C" if k.is_file() else ("R" if r.is_file() and r.stat().st_size > 0 else ""))
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("objects", nargs="*")
    ap.add_argument("--summary", action="store_true", help="solo totales por columna")
    a = ap.parse_args()
    data = ROOT / "data"
    dirs = [data / o for o in a.objects] if a.objects else sorted(
        p for p in data.iterdir() if p.is_dir() and not p.name.startswith(("_", ".")))
    rows = {d.name: checks(d) for d in dirs}
    if not rows:
        raise SystemExit(f"No hay objetos en {data}")
    cols = list(next(iter(rows.values())))
    width = max(len(n) for n in rows)
    print(f"{'objeto':<{width}}  " + " ".join(f"{c[:6]:>6}" for c in cols))
    if not a.summary:
        for name, c in rows.items():
            cells = [("x" if v else ".") if isinstance(v, bool) else (v or ".") for v in c.values()]
            print(f"{name:<{width}}  " + " ".join(f"{v:>6}" for v in cells))
        print()
    print(f"{'TOTAL':<{width}}  " + " ".join(f"{sum(bool(r[c]) for r in rows.values()):>6}" for c in cols)
          + f"   (de {len(rows)})")
    print("Técnicas: R=export crudo, C=limpio, E=evaluado. Etapas: x=hecho, .=falta")


if __name__ == "__main__":
    main()
