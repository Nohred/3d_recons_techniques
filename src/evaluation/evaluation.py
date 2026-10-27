import numpy as np
import open3d as o3d

# Importamos la función que creamos en el paso anterior
# (Asume que el archivo se llama registration.py)
from registration import register_point_clouds

def compute_metrics(source_pcd, target_gt_pcd, fscore_threshold_ratio=0.01):
    """
    Calcula las 5 métricas principales entre dos nubes de puntos YA ALINEADAS.
    
    Parámetros:
    - source_pcd: Nube reconstruida/limpia.
    - target_gt_pcd: Nube Ground Truth.
    - fscore_threshold_ratio: Tolerancia para el F-Score (ej. 0.01 = 1% de la diagonal).
    """
    print("\n[ Evaluando Métricas... ]")
    
    # 1. Calcular distancias mutuas al vecino más cercano (K-Nearest Neighbors interno de O3D)
    dists_source_to_gt = np.asarray(source_pcd.compute_point_cloud_distance(target_gt_pcd))
    dists_gt_to_source = np.asarray(target_gt_pcd.compute_point_cloud_distance(source_pcd))
    
    # Manejo de escenas vacías para evitar crashes matemáticos
    if len(dists_source_to_gt) == 0 or len(dists_gt_to_source) == 0:
        raise ValueError("Una de las nubes está vacía. Revisa la carga de archivos.")

    # 2. Distancia de Chamfer (L1 - Promedio)
    chamfer_dist = (np.mean(dists_source_to_gt) + np.mean(dists_gt_to_source)) / 2.0
    
    # 3. RMSE (Root Mean Square Error)
    rmse = np.sqrt((np.mean(dists_source_to_gt**2) + np.mean(dists_gt_to_source**2)) / 2.0)
    
    # 4. Distancia de Hausdorff (El peor error)
    hausdorff_dist = max(np.max(dists_source_to_gt), np.max(dists_gt_to_source))
    
    # 5. F-Score
    # Usamos la diagonal del GT para definir qué significa "estar cerca"
    gt_bbox = target_gt_pcd.get_axis_aligned_bounding_box()
    diag_target = np.linalg.norm(gt_bbox.get_extent())
    
    threshold = diag_target * fscore_threshold_ratio
    
    precision = np.sum(dists_source_to_gt < threshold) / len(dists_source_to_gt)
    recall = np.sum(dists_gt_to_source < threshold) / len(dists_gt_to_source)
    
    if (precision + recall) > 0:
        f_score = (2 * precision * recall) / (precision + recall)
    else:
        f_score = 0.0

    # 6. Geometric PSNR
    # Relación entre el tamaño de la caja (peak signal) y el error RMS (noise)
    if rmse > 0:
        psnr = 20 * np.log10(diag_target / rmse)
    else:
        psnr = float('inf') # Nubes idénticas sin error

    # Empaquetar resultados
    report = {
        "chamfer": chamfer_dist,
        "rmse": rmse,
        "hausdorff": hausdorff_dist,
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "psnr_db": psnr,
        "threshold": threshold
    }
    
    return report

def print_report(metrics):
    print("\n" + "="*45)
    print("      REPORTE DE MÉTRICAS GEOMÉTRICAS")
    print("="*45)
    print(f" Distancia Chamfer   : {metrics['chamfer']:.6f}")
    print(f" RMSE                : {metrics['rmse']:.6f}")
    print(f" Distancia Hausdorff : {metrics['hausdorff']:.6f}")
    print("-" * 45)
    print(f" F-Score (umbral {metrics['threshold']:.4f})")
    print(f"   -> Precisión      : {metrics['precision']*100:.2f}%")
    print(f"   -> Recall         : {metrics['recall']*100:.2f}%")
    print(f"   -> F-Score Total  : {metrics['f_score']*100:.2f}%")
    print("-" * 45)
    print(f" PSNR                : {metrics['psnr_db']:.2f} dB")
    print("="*45 + "\n")


if __name__ == "__main__":
    # 1. Definir rutas (Hardcoded para tu pipeline actual)
    source_path = "splat_clean.ply"
    gt_path = "splat_clean_manual.ply"
    
    print(f"Cargando {source_path} y {gt_path}...")
    source_pcd = o3d.io.read_point_cloud(source_path)
    gt_pcd = o3d.io.read_point_cloud(gt_path)
    
    # 2. Alinear las nubes usando nuestro módulo anterior
    # Esto garantiza que escala, rotación y traslación empaten antes de medir
    aligned_source, icp_info = register_point_clouds(source_pcd, gt_pcd)
    
    # 3. Calcular las 5 métricas de la nube alineada contra el Ground Truth
    # Pasamos 0.02 (2% del tamaño) como tolerancia razonable para el F-Score
    metrics_result = compute_metrics(aligned_source, gt_pcd, fscore_threshold_ratio=0.02)
    
    # 4. Imprimir la tabla de resultados
    print_report(metrics_result)
    
    # (Opcional) Guardar la nube alineada para inspección visual
    o3d.io.write_point_cloud("splat_aligned.ply", aligned_source)