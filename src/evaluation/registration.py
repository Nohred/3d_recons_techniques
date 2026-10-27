import copy
import numpy as np
import open3d as o3d

def register_point_clouds(source_pcd, target_gt_pcd):
    """
    Alinea una nube de puntos (source) hacia un Ground Truth (target).
    Resuelve Traslación, Rotación y Escala.
    
    Retorna:
    - source_aligned: La nube original transformada y lista para métricas.
    - result_icp: Objeto con los datos del ajuste (fitness, RMSE, matriz).
    """
    # Trabajar sobre copias para no alterar los objetos originales
    source = copy.deepcopy(source_pcd)
    target = copy.deepcopy(target_gt_pcd)

    print("--- Iniciando Registro de Nubes de Puntos ---")

    # --- 1. CENTRADO Y ESCALA APROXIMADA ---
    # Mover al mismo centro
    c_src = source.get_center()
    c_tgt = target.get_center()
    source.translate(c_tgt - c_src)

    # Calcular dispersión (escala) basada en la distancia media al centroide
    pts_src = np.asarray(source.points)
    pts_tgt = np.asarray(target.points)
    
    scale_src = np.mean(np.linalg.norm(pts_src - c_tgt, axis=1))
    scale_tgt = np.mean(np.linalg.norm(pts_tgt - c_tgt, axis=1))
    
    initial_scale = scale_tgt / (scale_src + 1e-8)
    source.scale(initial_scale, center=c_tgt)
    print(f"1. Escala aproximada aplicada (Factor: {initial_scale:.4f})")

    # --- 2. PREPARACIÓN Y EXTRACCIÓN DE CARACTERÍSTICAS ---
    # Usamos el tamaño del GT para dictar la resolución del cálculo
    diag = np.linalg.norm(target.get_axis_aligned_bounding_box().get_extent())
    voxel_size = diag * 0.02  # Voxel del 2% del tamaño total
    
    src_down = source.voxel_down_sample(voxel_size)
    tgt_down = target.voxel_down_sample(voxel_size)
    
    # Estimar normales (necesario para FPFH)
    src_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    tgt_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    # FPFH (Fast Point Feature Histograms) - Geometría local
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))

    # --- 3. ALINEACIÓN GLOBAL (RANSAC) ---
    print("2. Ejecutando alineación global (RANSAC)...")
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, True, voxel_size * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    # Aplicar la matriz de rotación/traslación inicial a la nube original densa
    source.transform(result_ransac.transformation)

    # --- 4. ALINEACIÓN FINA (ICP CON ESCALA) ---
    print("3. Ejecutando ajuste fino milimétrico (ICP con escala)...")
    icp_radius = diag * 0.05
    
    # La magia ocurre aquí: le decimos al ICP que sí tiene permitido deformar la escala
    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, icp_radius, np.identity(4),
        estimation_method,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300)
    )
    
    # Aplicar la matriz perfecta final
    source.transform(result_icp.transformation)

    print("--- Registro Completado ---")
    print(f"-> Fitness (Puntos emparejados): {result_icp.fitness * 100:.2f}%")
    print(f"-> RMSE (Error promedio inliers): {result_icp.inlier_rmse:.5f}")

    return source, result_icp