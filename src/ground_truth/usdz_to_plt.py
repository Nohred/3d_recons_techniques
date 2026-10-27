"""Genera un Ground Truth PLY por objeto bajo data/<object_id>/ground_truth/."""
from __future__ import annotations

import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ================= CONFIGURACION EDITABLE =================
OBJETOS = None  # None: todos los data/<object_id> con assets/original.usdz
PUNTOS_DESEADOS = 200_000
SOBRESCRIBIR = False
# ==========================================================


def object_directories() -> list[Path]:
    data_root = ROOT / "data"
    selected = (
        sorted(path for path in data_root.iterdir()
               if path.is_dir() and (path / "assets" / "original.usdz").is_file())
        if OBJETOS is None
        else [data_root / object_id for object_id in OBJETOS]
    )
    if not selected:
        raise RuntimeError("No hay objetos canónicos con assets/original.usdz.")
    for path in selected:
        if not (path / "assets" / "original.usdz").is_file():
            raise FileNotFoundError(f"Falta el USDZ canónico: {path}")
    return selected


async def export_usd_to_glb(usdz_path: Path, glb_path: Path) -> None:
    import omni.kit.asset_converter

    context = omni.kit.asset_converter.AssetConverterContext()
    context.ignore_materials = False
    context.ignore_animations = True
    task = omni.kit.asset_converter.get_instance().create_converter_task(
        str(usdz_path), str(glb_path), None, context
    )
    if not await task.wait_until_finished():
        raise RuntimeError(f"No se pudo convertir {usdz_path} a GLB.")


def convert_mesh_to_pointcloud(mesh_path: Path, ply_path: Path) -> None:
    import numpy as np
    import open3d as o3d
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    if len(mesh.faces) == 0:
        raise RuntimeError(f"La malla convertida está vacía: {mesh_path}")
    if hasattr(mesh.visual, "to_color"):
        mesh.visual = mesh.visual.to_color()
    points, faces = trimesh.sample.sample_surface(mesh, PUNTOS_DESEADOS)
    if mesh.visual.kind == "face":
        colors = mesh.visual.face_colors[faces][:, :3]
    elif mesh.visual.kind == "vertex":
        colors = mesh.visual.vertex_colors[mesh.faces[faces, 0]][:, :3]
    else:
        colors = np.full((len(points), 3), 150, dtype=np.uint8)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float) / 255.0)
    cloud.estimate_normals()
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(ply_path), cloud):
        raise RuntimeError(f"No se pudo escribir el Ground Truth: {ply_path}")

### Legacy

# async def export_usd_to_glb(usdz_path, glb_path):
#     print(f"[1/3] Extrayendo geometría y texturas de {usdz_path} a formato GLB...")
    
#     converter_context = omni.kit.asset_converter.AssetConverterContext()
#     # CAMBIO CLAVE: Permitir materiales para retener las texturas
#     converter_context.ignore_materials = False 
#     converter_context.ignore_animations = True
    
#     instance = omni.kit.asset_converter.get_instance()
#     task = instance.create_converter_task(usdz_path, glb_path, None, converter_context)
    
#     success = await task.wait_until_finished()
#     return success

# def convert_mesh_to_colored_pointcloud(mesh_path, ply_path, num_points=2000000):
#     print(f"[2/3] Cargando malla y procesando texturas...")
    
#     # Cargar con trimesh para poder procesar mapas UV y materiales
#     mesh = trimesh.load(mesh_path, force='mesh', process=False)
    
#     if len(mesh.faces) == 0:
#         print("Error: La malla exportada está vacía.")
#         return False
        
#     # Convertir el mapa de textura (UV) a colores explícitos mapeados a la geometría
#     if hasattr(mesh.visual, 'to_color'):
#         mesh.visual = mesh.visual.to_color()
    
#     print(f"[3/3] Generando nube de {num_points} puntos con extracción de color...")
#     # sample_surface retorna las coordenadas XYZ y el índice de la cara (triángulo) a la que pertenece
#     points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    
#     # Extraer el color correspondiente a cada cara muestreada
#     if mesh.visual.kind == 'face':
#         colors = mesh.visual.face_colors[face_indices][:, :3]
#     elif mesh.visual.kind == 'vertex':
#         faces = mesh.faces[face_indices]
#         colors = mesh.visual.vertex_colors[faces[:, 0]][:, :3]
#     else:
#         # Fallback: Color gris si el modelo original no tenía textura ni material
#         colors = np.ones((num_points, 3)) * 150 
        
#     # Normalizar colores al rango [0, 1] que exige Open3D para los archivos .ply
#     colors = colors / 255.0
    
#     # Ensamblar y guardar la nube de puntos final con Open3D
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
    
#     # Computar normales por si las necesitas en métricas como Chamfer distance o F-Score
#     pcd.estimate_normals()
    
#     o3d.io.write_point_cloud(ply_path, pcd)
    # return True


async def main() -> None:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        for object_dir in object_directories():
            output = object_dir / "ground_truth" / "ground_truth.ply"
            if output.exists() and not SOBRESCRIBIR:
                print(f"OMITIDO (ya existe): {output}")
                continue
            temporary = object_dir / "ground_truth" / ".temporary_model.glb"
            try:
                await export_usd_to_glb(
                    object_dir / "assets" / "original.usdz", temporary
                )
                convert_mesh_to_pointcloud(temporary, output)
                print(f"GT generado: {output}")
            finally:
                temporary.unlink(missing_ok=True)
                for sidecar in temporary.parent.glob(f"{temporary.stem}.*"):
                    if sidecar != output:
                        sidecar.unlink(missing_ok=True)
    finally:
        app.close()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
