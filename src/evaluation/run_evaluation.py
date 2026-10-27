"""Registra y evalúa reconstrucciones contra el GT de cada objeto."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ================= CONFIGURACION EDITABLE =================
OBJETOS = None  # None: objetos con GT y export Splatfacto limpio
UMBRAL_FSCORE = 0.02
SOBRESCRIBIR_REPORTES = False
PROCESAR_NERFACTO = False  # Reservado hasta tener limpiador específico.
# ==========================================================


def object_directories() -> list[Path]:
    data_root = ROOT / "data"
    if OBJETOS is None:
        return sorted(
            path for path in data_root.iterdir()
            if path.is_dir()
            and (path / "ground_truth" / "ground_truth.ply").is_file()
            and (path / "reconstruction" / "exports" / "splatfacto"
                 / "gaussian_centers_clean.ply").is_file()
        )
    return [data_root / object_id for object_id in OBJETOS]


def evaluate_object(object_dir: Path, reports_dir: Path) -> dict:
    import open3d as o3d

    from src.evaluation.evaluation import compute_metrics
    from src.evaluation.registration import register_point_clouds

    object_id = object_dir.name
    source_path = (
        object_dir / "reconstruction" / "exports" / "splatfacto"
        / "gaussian_centers_clean.ply"
    )
    gt_path = object_dir / "ground_truth" / "ground_truth.ply"
    report_path = reports_dir / f"{object_id}.json"
    if report_path.exists() and not SOBRESCRIBIR_REPORTES:
        raise FileExistsError(
            f"Ya existe {report_path}; activa SOBRESCRIBIR_REPORTES para reemplazarlo."
        )
    source = o3d.io.read_point_cloud(str(source_path))
    target = o3d.io.read_point_cloud(str(gt_path))
    if source.is_empty() or target.is_empty():
        raise RuntimeError(f"Nube vacía para {object_id}: {source_path}, {gt_path}")
    aligned, registration = register_point_clouds(source, target)
    metrics = compute_metrics(aligned, target, UMBRAL_FSCORE)
    result = {
        "object_id": object_id,
        "method": "splatfacto",
        "source": str(source_path.relative_to(ROOT)),
        "ground_truth": str(gt_path.relative_to(ROOT)),
        "source_points": len(source.points),
        "ground_truth_points": len(target.points),
        "registration": {
            "fitness": float(registration.fitness),
            "inlier_rmse": float(registration.inlier_rmse),
            "transformation": registration.transformation.tolist(),
        },
        "metrics": {key: float(value) for key, value in metrics.items()},
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> None:
    reports_dir = ROOT / "reports"
    objects = object_directories()
    if not objects:
        raise RuntimeError(
            "No hay objetos con GT y gaussian_centers_clean.ply. "
            "Ejecuta usdz_to_plt.py y clean_reconstructions.py."
        )
    if PROCESAR_NERFACTO:
        raise RuntimeError(
            "Nerfacto está deshabilitado: falta su script de limpieza específico."
        )
    results = []
    failures = []
    for object_dir in objects:
        try:
            results.append(evaluate_object(object_dir, reports_dir))
        except Exception as error:
            failures.append({"object_id": object_dir.name, "error": str(error)})
            traceback.print_exc()
    summary = {
        "objects_requested": [path.name for path in objects],
        "objects_evaluated": len(results),
        "failures": failures,
        "method": "splatfacto",
        "fscore_threshold_ratio": UMBRAL_FSCORE,
        "results": results,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"Fallaron objetos: {[item['object_id'] for item in failures]}")


if __name__ == "__main__":
    main()
