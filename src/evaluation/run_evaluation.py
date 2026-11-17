"""Registra y evalúa cada técnica limpia contra el GT de cada objeto.

Uso: python src/evaluation/run_evaluation.py [objeto ...] [--overwrite] [--methods a,b]
Salidas:
  reports/evaluation/<objeto>__<metodo>.json   (uno por par; se reutiliza si existe)
  reports/evaluation/summary.json / summary.csv (tabla comparativa)
"""
from __future__ import annotations

from pathlib import Path
import csv
import json
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ================= CONFIGURACION EDITABLE =================
OBJETOS = None  # None: todos los objetos con ground_truth/ground_truth.ply
UMBRAL_FSCORE = 0.02
SOBRESCRIBIR_REPORTES = False
METODOS = None  # None: todas las técnicas; o "splatfacto,photogrammetry"
# ==========================================================

REPORTS_DIR = ROOT / "reports" / "evaluation"


def gt_path(object_dir: Path) -> Path:
    return object_dir / "ground_truth" / "ground_truth.ply"


def object_directories(objects) -> list[Path]:
    data_root = ROOT / "data"
    if objects is None:
        return sorted(p for p in data_root.iterdir() if p.is_dir() and gt_path(p).is_file())
    return [data_root / object_id for object_id in objects]


def evaluate(object_dir: Path, method: str, source_path: Path) -> dict:
    import open3d as o3d

    from src.evaluation.evaluation import compute_metrics
    from src.evaluation.registration import register_point_clouds

    source = o3d.io.read_point_cloud(str(source_path))
    target = o3d.io.read_point_cloud(str(gt_path(object_dir)))
    if source.is_empty() or target.is_empty():
        raise RuntimeError(f"Nube vacía: {source_path} o {gt_path(object_dir)}")
    aligned, registration = register_point_clouds(source, target)
    metrics = compute_metrics(aligned, target, UMBRAL_FSCORE)
    return {
        "object_id": object_dir.name,
        "method": method,
        "source": str(source_path.relative_to(ROOT)),
        "ground_truth": str(gt_path(object_dir).relative_to(ROOT)),
        "source_points": len(source.points),
        "ground_truth_points": len(target.points),
        "fscore_threshold_ratio": UMBRAL_FSCORE,
        "registration": {
            "fitness": float(registration.fitness),
            "inlier_rmse": float(registration.inlier_rmse),
            "transformation": registration.transformation.tolist(),
        },
        "metrics": {key: float(value) for key, value in metrics.items()},
    }


def write_summary(results: list[dict], failures: list[dict], methods: list[str]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "methods": methods,
        "fscore_threshold_ratio": UMBRAL_FSCORE,
        "evaluated": len(results),
        "failures": failures,
        "results": results,
    }
    (REPORTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    metric_keys = sorted({k for r in results for k in r["metrics"]})
    with (REPORTS_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["object_id", "method", "source_points", "fitness", "inlier_rmse", *metric_keys])
        for r in sorted(results, key=lambda r: (r["object_id"], r["method"])):
            writer.writerow([r["object_id"], r["method"], r["source_points"],
                             r["registration"]["fitness"], r["registration"]["inlier_rmse"],
                             *[r["metrics"].get(k, "") for k in metric_keys]])


def main() -> None:
    from src.cli_utils import parse_cli, parse_methods_arg
    from src.evaluation.methods import clean_path, parse_methods

    objects, overwrite = parse_cli(OBJETOS, SOBRESCRIBIR_REPORTES, __doc__)
    methods = parse_methods(parse_methods_arg() or METODOS)
    object_dirs = object_directories(objects)
    if not object_dirs:
        raise SystemExit("No hay objetos con ground_truth/ground_truth.ply. Ejecuta: ./run.sh gt")

    results, failures, missing = [], [], 0
    for object_dir in object_dirs:
        if not gt_path(object_dir).is_file():
            failures.append({"object_id": object_dir.name, "method": "*", "error": "falta GT"})
            continue
        for method in methods:
            source = clean_path(object_dir, method)
            if not source.is_file():
                missing += 1
                continue
            report = REPORTS_DIR / f"{object_dir.name}__{method}.json"
            if report.is_file() and not overwrite:
                results.append(json.loads(report.read_text(encoding="utf-8")))
                continue
            print(f"=== {object_dir.name} :: {method} ===", flush=True)
            try:
                result = evaluate(object_dir, method, source)
            except Exception as error:
                traceback.print_exc()
                failures.append({"object_id": object_dir.name, "method": method, "error": str(error)})
                continue
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            results.append(result)

    write_summary(results, failures, methods)
    print(f"Evaluados: {len(results)} | sin nube limpia: {missing} | fallos: {len(failures)}")
    print(f"Resumen: {REPORTS_DIR / 'summary.csv'}")
    if failures:
        raise SystemExit(f"Fallaron: {[(f['object_id'], f['method']) for f in failures]}")


if __name__ == "__main__":
    main()
