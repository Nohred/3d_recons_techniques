# IMPORTANTE: SimulationApp debe iniciarse antes de importar cualquier módulo de omni o pxr
import isaacsim
from isaacsim.simulation_app import SimulationApp
simulation_app = SimulationApp({"headless": True})

import asyncio
from isaacsim.core.utils.extensions import enable_extension
enable_extension("omni.kit.asset_converter")
import omni.kit.asset_converter
import open3d as o3d
import trimesh
import numpy as np
import os

async def export_usd_to_glb(usdz_path, glb_path):
    print(f"[1/3] Extrayendo geometría y texturas de {usdz_path} a formato GLB...")
    
    converter_context = omni.kit.asset_converter.AssetConverterContext()
    # CAMBIO CLAVE: Permitir materiales para retener las texturas
    converter_context.ignore_materials = False 
    converter_context.ignore_animations = True
    
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(usdz_path, glb_path, None, converter_context)
    
    success = await task.wait_until_finished()
    return success

def convert_mesh_to_colored_pointcloud(mesh_path, ply_path, num_points=2000000):
    print(f"[2/3] Cargando malla y procesando texturas...")
    
    # Cargar con trimesh para poder procesar mapas UV y materiales
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    
    if len(mesh.faces) == 0:
        print("Error: La malla exportada está vacía.")
        return False
        
    # Convertir el mapa de textura (UV) a colores explícitos mapeados a la geometría
    if hasattr(mesh.visual, 'to_color'):
        mesh.visual = mesh.visual.to_color()
    
    print(f"[3/3] Generando nube de {num_points} puntos con extracción de color...")
    # sample_surface retorna las coordenadas XYZ y el índice de la cara (triángulo) a la que pertenece
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    
    # Extraer el color correspondiente a cada cara muestreada
    if mesh.visual.kind == 'face':
        colors = mesh.visual.face_colors[face_indices][:, :3]
    elif mesh.visual.kind == 'vertex':
        faces = mesh.faces[face_indices]
        colors = mesh.visual.vertex_colors[faces[:, 0]][:, :3]
    else:
        # Fallback: Color gris si el modelo original no tenía textura ni material
        colors = np.ones((num_points, 3)) * 150 
        
    # Normalizar colores al rango [0, 1] que exige Open3D para los archivos .ply
    colors = colors / 255.0
    
    # Ensamblar y guardar la nube de puntos final con Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Computar normales por si las necesitas en métricas como Chamfer distance o F-Score
    pcd.estimate_normals()
    
    o3d.io.write_point_cloud(ply_path, pcd)
    return True

async def main():
    # --- CONFIGURACIÓN ---
    usdz_file = "/home/ccdatos/Escritorio/3D_reconstruction/data/sneaker/assets/original.usdz"   
    temp_mesh_file = "temp_mesh.glb" # Cambiado a GLB para incrustar texturas automáticamente
    final_ply_file = "ground_truth.ply"
    puntos_deseados = 2000000             
    # ---------------------

    if not os.path.exists(usdz_file):
        print(f"Error: No se encontró el archivo {usdz_file}")
        simulation_app.close()
        return

    export_success = await export_usd_to_glb(usdz_file, temp_mesh_file)
    
    if export_success:
        pcd_success = convert_mesh_to_colored_pointcloud(temp_mesh_file, final_ply_file, puntos_deseados)
        
        if pcd_success:
            print(f"\n¡Éxito! Ground Truth a color generado en: {final_ply_file}")
            
        # Limpieza del temporal sin preocuparse por archivos .mtl asociados
        if os.path.exists(temp_mesh_file):
            os.remove(temp_mesh_file)
    else:
        print("\nFallo en la extracción de la malla desde el USDZ.")

    simulation_app.close()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())