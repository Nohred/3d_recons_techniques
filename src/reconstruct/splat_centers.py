"""Convierte el PLY de gaussianas a un PLY XYZ+RGB de centros. Uso: python splat_centers.py entrada.ply salida.ply"""
import sys
import numpy as np
from plyfile import PlyData, PlyElement

entrada, salida = sys.argv[1], sys.argv[2]
datos = PlyData.read(entrada)["vertex"].data
campos = set(datos.dtype.names)
xyz = np.column_stack([datos[k] for k in ("x", "y", "z")]).astype(float)
if {"f_dc_0", "f_dc_1", "f_dc_2"} <= campos:
    dc = np.column_stack([datos[k] for k in ("f_dc_0", "f_dc_1", "f_dc_2")]).astype(float)
    rgb = np.clip(0.5 + 0.28209479177387814 * dc, 0, 1)
else:
    rgb = np.full(xyz.shape, 0.7)
ok = np.isfinite(xyz).all(axis=1) & np.isfinite(rgb).all(axis=1)
xyz, rgb = xyz[ok], rgb[ok]
if len(xyz) == 0:
    raise SystemExit("No hay centros validos")
v = np.empty(len(xyz), dtype=[("x","f4"),("y","f4"),("z","f4"),("red","u1"),("green","u1"),("blue","u1")])
for i, c in enumerate("xyz"):
    v[c] = xyz[:, i]
col = np.rint(rgb * 255).astype(np.uint8)
for i, c in enumerate(("red", "green", "blue")):
    v[c] = col[:, i]
PlyData([PlyElement.describe(v, "vertex")], text=False,
        comments=["Gaussian centers, NOT an extracted surface"]).write(salida)
print(f"Centros: {len(xyz)} de {len(datos)}")
