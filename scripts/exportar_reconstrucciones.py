from pathlib import Path
import json
import os
import shutil
import subprocess
import time

# Ejecutar desde el proyecto Pixi: pixi run python /ruta/a/este_script.py
PROYECTO_PIXI = Path.home() / "Escritorio/3D_reconstruction/nerfstudio"
CONFIG_NERF = Path.home() / "Escritorio/data_resul/data_proc/nerfacto/2026-10-19_143508/config.yml"
CONFIG_SPLAT = Path.home() / "Escritorio/data_resul/data_proc/splatfacto/2026-10-19_143948/config.yml"
SALIDA = Path.home() / "Escritorio/data_export/escena_01_auto"
PUNTOS_NERF = 500_000
RAYOS_POR_BATCH = 8192
EXPORTAR_NERF = True
EXPORTAR_SPLAT = True
# None conserva todos los centros finitos exportados; no es limpieza de superficie.
OPACIDAD_MINIMA_CENTROS = None

import numpy as np
from plyfile import PlyData, PlyElement


def exportar(nombre, argumentos):
    log = SALIDA / f"{nombre}.log"
    print(f"Exportando {nombre}. Registro: {log}", flush=True)
    with log.open("w", encoding="utf-8") as archivo:
        subprocess.run(["ns-export", *argumentos], cwd=PROYECTO_PIXI,
                       env=os.environ.copy(), stdout=archivo,
                       stderr=subprocess.STDOUT, check=True)


def convertir_centros(origen, destino):
    datos = PlyData.read(str(origen))["vertex"].data
    campos = set(datos.dtype.names)
    xyz = np.column_stack([datos[k] for k in ("x", "y", "z")])
    mascara = np.isfinite(xyz).all(axis=1)
    if {"red", "green", "blue"} <= campos:
        rgb = np.column_stack([datos[k] for k in ("red", "green", "blue")]).astype(float) / 255
    elif {"f_dc_0", "f_dc_1", "f_dc_2"} <= campos:
        dc = np.column_stack([datos[k] for k in ("f_dc_0", "f_dc_1", "f_dc_2")])
        rgb = np.clip(0.5 + 0.28209479177387814 * dc, 0, 1)
    else:
        rgb = np.full(xyz.shape, 0.7)
    mascara &= np.isfinite(rgb).all(axis=1)
    if OPACIDAD_MINIMA_CENTROS is not None:
        if "opacity" not in campos:
            raise RuntimeError("No existe el campo opacity.")
        alpha = 1 / (1 + np.exp(-np.clip(datos["opacity"].astype(float), -60, 60)))
        mascara &= alpha >= OPACIDAD_MINIMA_CENTROS
    xyz, rgb = xyz[mascara], rgb[mascara]
    if len(xyz) == 0:
        raise RuntimeError("No quedaron centros validos.")
    vertices = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, campo in enumerate(("x", "y", "z")):
        vertices[campo] = xyz[:, i]
    colores = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    for i, campo in enumerate(("red", "green", "blue")):
        vertices[campo] = colores[:, i]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False,
            comments=["Gaussian centers, NOT an extracted object surface",
                      "RGB from SH DC only when RGB fields are absent"]).write(str(destino))
    return {"centros_entrada": len(datos), "centros_salida": len(xyz),
            "opacidad_minima": OPACIDAD_MINIMA_CENTROS,
            "advertencia": "Centros de gaussianas; no equivalen a una superficie reconstruida."}


def main():
    if not PROYECTO_PIXI.is_dir() or not shutil.which("ns-export"):
        raise RuntimeError("Ejecuta con pixi run python desde el proyecto de Nerfstudio.")
    for activo, config in [(EXPORTAR_NERF, CONFIG_NERF), (EXPORTAR_SPLAT, CONFIG_SPLAT)]:
        if activo and not config.is_file():
            raise FileNotFoundError(config)
    if SALIDA.exists():
        raise RuntimeError(f"Ya existe {SALIDA}. Cambia SALIDA; no se sobrescribe.")
    SALIDA.mkdir(parents=True)
    resumen = {"config_nerf": str(CONFIG_NERF), "config_splat": str(CONFIG_SPLAT),
               "espacio": "Coordenadas internas del modelo; NO asumir metros de Isaac Sim."}
    inicio = time.perf_counter()
    if EXPORTAR_NERF:
        carpeta = SALIDA / "nerfacto_scene"
        exportar("nerfacto", ["pointcloud", "--load-config", str(CONFIG_NERF),
                 "--output-dir", str(carpeta), "--num-points", str(PUNTOS_NERF),
                 "--normal-method", "open3d", "--remove-outliers", "False",
                 "--num-rays-per-batch", str(RAYOS_POR_BATCH)])
        nube = carpeta / "point_cloud.ply"
        if not nube.is_file() or nube.stat().st_size == 0:
            raise RuntimeError("No se genero point_cloud.ply de Nerfacto.")
        resumen["nerf_ply"] = str(nube)
    if EXPORTAR_SPLAT:
        carpeta = SALIDA / "splatfacto_native"
        exportar("splatfacto", ["gaussian-splat", "--load-config", str(CONFIG_SPLAT),
                 "--output-dir", str(carpeta)])
        archivos = list(carpeta.glob("*.ply"))
        if len(archivos) != 1:
            raise RuntimeError(f"Se esperaba un PLY de gaussianas, se encontraron {archivos}")
        destino = SALIDA / "splatfacto_centers_NOT_SURFACE.ply"
        resumen["splat_conversion"] = convertir_centros(archivos[0], destino)
        resumen["splat_native"] = str(archivos[0])
        resumen["splat_centers"] = str(destino)
    resumen["duracion_s"] = time.perf_counter() - inicio
    (SALIDA / "export_report.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Exportacion terminada: {SALIDA}", flush=True)


if __name__ == "__main__":
    main()
