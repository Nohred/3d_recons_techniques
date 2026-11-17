"""Limpia los exports de cada técnica (splatfacto, nerfacto, photogrammetry, rgbd_fusion).

Uso: python src/evaluation/clean_reconstructions.py [objeto ...] [--overwrite] [--methods a,b]
Parámetros por técnica en src/evaluation/methods.py. Un fallo no detiene el lote.
"""
from __future__ import annotations

from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ================= CONFIGURACION EDITABLE =================
OBJETOS = None  # None: todos los data/<object_id> con algún export disponible
SOBRESCRIBIR = False
METODOS = None  # None: todas las técnicas de methods.METHODS; o "splatfacto,nerfacto"
# ==========================================================


def object_directories(objects) -> list[Path]:
    data_root = ROOT / "data"
    if objects is None:
        return sorted(p for p in data_root.iterdir()
                      if p.is_dir() and (p / "reconstruction" / "exports").is_dir())
    return [data_root / object_id for object_id in objects]


def main() -> None:
    from src.cli_utils import parse_cli, parse_methods_arg
    from src.evaluation.clean_pointcloud import clean_point_cloud
    from src.evaluation.methods import METHODS, clean_path, parse_methods, raw_path

    objects, overwrite = parse_cli(OBJETOS, SOBRESCRIBIR, __doc__)
    methods = parse_methods(parse_methods_arg() or METODOS)
    counts = {"ok": 0, "skipped": 0, "missing": 0}
    failures = []
    for object_dir in object_directories(objects):
        for method in methods:
            source, destination = raw_path(object_dir, method), clean_path(object_dir, method)
            if not source.is_file():
                counts["missing"] += 1
                continue
            if destination.exists() and not overwrite:
                counts["skipped"] += 1
                continue
            print(f"=== {object_dir.name} :: {method} ===", flush=True)
            try:
                clean_point_cloud(str(source), str(destination), **METHODS[method]["params"])
                counts["ok"] += 1
            except (Exception, SystemExit):
                traceback.print_exc()
                destination.unlink(missing_ok=True)
                failures.append(f"{object_dir.name}:{method}")
    print(f"Limpieza: {counts['ok']} nuevas, {counts['skipped']} existentes omitidas, "
          f"{counts['missing']} sin export.")
    if failures:
        raise SystemExit(f"Fallaron: {failures}")


if __name__ == "__main__":
    main()
