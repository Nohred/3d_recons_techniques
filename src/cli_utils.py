"""CLI común: objetos posicionales y --overwrite sobre los valores editables del script.

Usa parse_known_args para tolerar argumentos extra que Isaac Sim/Kit pueda recibir.
"""
from __future__ import annotations

import argparse


def parse_cli(default_objects, default_overwrite: bool, description: str | None = None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "objects", nargs="*",
        help="object_id a procesar (sin argumentos: valor OBJETOS del script)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="rehacer salidas existentes (por defecto se omiten)",
    )
    parser.add_argument(
        "--methods", default=None,
        help="solo limpieza/evaluación: técnicas separadas por coma "
             "(splatfacto,nerfacto,photogrammetry,rgbd_fusion)",
    )
    args, _unknown = parser.parse_known_args()
    objects = [o.removesuffix(".usdz") for o in args.objects] or default_objects
    return objects, bool(args.overwrite or default_overwrite)


def parse_methods_arg() -> str | None:
    """Lee --methods a,b,c (técnicas de reconstrucción) sin interferir con parse_cli."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--methods", default=None)
    args, _unknown = parser.parse_known_args()
    return args.methods
