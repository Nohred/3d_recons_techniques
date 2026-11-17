"""Registro único de técnicas de reconstrucción: export crudo, nube limpia y parámetros de limpieza.

Lo usan clean_reconstructions.py, run_evaluation.py y scripts/status.py.
Rutas relativas a data/<object_id>/reconstruction/exports/.
"""
from __future__ import annotations

from pathlib import Path

METHODS: dict[str, dict] = {
    "splatfacto": {
        "raw": "splatfacto/gaussian_centers_NOT_SURFACE.ply",
        "clean": "splatfacto/gaussian_centers_clean.ply",
        "params": dict(min_opacity=0.3, crop_ratio=0.035, floor_thr=0.015,
                       cluster_eps=0.08, cluster_min=50, remove_floor=True),
    },
    # Parámetros genéricos: pendiente ajustar un limpiador específico para Nerfacto.
    "nerfacto": {
        "raw": "nerfacto/point_cloud.ply",
        "clean": "nerfacto/point_cloud_clean.ply",
        "params": dict(min_opacity=0.0, crop_ratio=0.035, floor_thr=0.015,
                       cluster_eps=0.08, cluster_min=50, remove_floor=True),
    },
    "photogrammetry": {
        "raw": "photogrammetry/fused.ply",
        "clean": "photogrammetry/fused_clean.ply",
        "params": dict(min_opacity=0.0, crop_ratio=0.035, floor_thr=0.015,
                       cluster_eps=0.08, cluster_min=50, remove_floor=True),
    },
    # rgbd_fusion.py ya recorta suelo y entorno con la geometría conocida del tablero:
    # aquí solo se quitan islas y outliers.
    "rgbd_fusion": {
        "raw": "rgbd_fusion/fused.ply",
        "clean": "rgbd_fusion/fused_clean.ply",
        "params": dict(min_opacity=0.0, crop_ratio=0.0, remove_floor=False,
                       cluster_eps=0.05, cluster_min=30),
    },
}


def exports_dir(object_dir: Path) -> Path:
    return object_dir / "reconstruction" / "exports"


def raw_path(object_dir: Path, method: str) -> Path:
    return exports_dir(object_dir) / METHODS[method]["raw"]


def clean_path(object_dir: Path, method: str) -> Path:
    return exports_dir(object_dir) / METHODS[method]["clean"]


def parse_methods(value: str | None) -> list[str]:
    """'splatfacto,nerfacto' -> lista validada; None/'' -> todas."""
    if not value:
        return list(METHODS)
    methods = [m.strip() for m in value.split(",") if m.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Métodos desconocidos: {unknown}; válidos: {list(METHODS)}")
    return methods
