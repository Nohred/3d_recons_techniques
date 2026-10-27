"""Limpia exports de reconstrucción sin procesar todavía Nerfacto."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ================= CONFIGURACION EDITABLE =================
OBJETOS = None  # None: todos los data/<object_id> con exports disponibles
SOBRESCRIBIR = False
MIN_OPACIDAD = 0.3
CROP_RATIO = 0.035
FLOOR_THR = 0.015
CLUSTER_EPS = 0.08
CLUSTER_MIN = 50
# ==========================================================


def object_directories() -> list[Path]:
    data_root = ROOT / "data"
    if OBJETOS is None:
        result = sorted(
            path for path in data_root.iterdir()
            if path.is_dir()
            and (path / "reconstruction" / "exports" / "splatfacto"
                 / "gaussian_centers_NOT_SURFACE.ply").is_file()
        )
    else:
        result = [data_root / object_id for object_id in OBJETOS]
    if not result:
        raise RuntimeError("No hay exports gaussian_centers de Splatfacto.")
    return result


def clean_splatfacto(object_dir: Path) -> Path:
    from evaluation.clean_splatting import clean_point_cloud

    source = (
        object_dir / "reconstruction" / "exports" / "splatfacto"
        / "gaussian_centers_NOT_SURFACE.ply"
    )
    destination = (
        object_dir / "reconstruction" / "exports" / "splatfacto"
        / "gaussian_centers_clean.ply"
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and not SOBRESCRIBIR:
        print(f"OMITIDO (ya existe): {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    clean_point_cloud(
        inp_path=str(source),
        out_path=str(destination),
        min_opacity=MIN_OPACIDAD,
        crop_ratio=CROP_RATIO,
        floor_thr=FLOOR_THR,
        cluster_eps=CLUSTER_EPS,
        cluster_min=CLUSTER_MIN,
        remove_floor=True,
    )
    return destination


def main() -> None:
    for object_dir in object_directories():
        cleaned = clean_splatfacto(object_dir)
        print(f"Splatfacto limpio: {cleaned}")

        # Preparación futura de Nerfacto:
        # nerfacto_source = (
        #     object_dir / "reconstruction" / "exports" / "nerfacto"
        #     / "point_cloud.ply"
        # )
        # Cuando exista su limpiador específico, procesarlo aquí sin reutilizar
        # automáticamente los parámetros de Splatfacto.


if __name__ == "__main__":
    main()
