import argparse
import numpy as np
import open3d as o3d
from plyfile import PlyData

SH_C0 = 0.28209479177387814

def load_pcd(path, min_opacity=0.3):
    v = PlyData.read(path)["vertex"]
    names = v.data.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    
    if "f_dc_0" in names:
        f = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)
        rgb = np.clip(0.5 + SH_C0 * f, 0, 1)
    elif "red" in names:
        rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float64)
        rgb = rgb / 255.0 if rgb.max() > 1 else rgb
    else:
        rgb = np.full_like(xyz, 0.7)
    
    if "opacity" in names and min_opacity > 0:
        alpha = 1 / (1 + np.exp(-np.asarray(v["opacity"], dtype=np.float64)))
        m = alpha > min_opacity
        print(f"opacidad > {min_opacity}: {m.sum():,}/{len(m):,} gaussianas")
        xyz, rgb = xyz[m], rgb[m]
        
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(xyz)
    p.colors = o3d.utility.Vector3dVector(rgb)
    return p


def clean_point_cloud(
    inp_path, 
    out_path, 
    min_opacity=0.3, 
    crop_ratio=0.035, 
    floor_thr=0.015, 
    cluster_eps=0.08,
    cluster_min=50,
    remove_floor=True
):
    # --- 1. CARGA DE DATOS ---
    pcd = load_pcd(inp_path, min_opacity)
    n0 = len(pcd.points)
    diag_orig = np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent())
    print(f"Original: {n0:,} puntos | Diagonal: {diag_orig:.3f}")

    # --- 2. RECORTAR POR RADIO (CROPPING) ---
    if crop_ratio > 0:
        pts = np.asarray(pcd.points)
        center = np.median(pts, axis=0) 
        dists = np.linalg.norm(pts - center, axis=1)
        
        limit = diag_orig * crop_ratio
        print(f"Aplicando machetazo esférico (radio: {limit:.3f})...")
        mask = dists < limit
        pcd = pcd.select_by_index(np.where(mask)[0])
        print(f"-> Quedan {len(pcd.points):,} puntos tras eliminar exteriores.")

    # --- 2.5 ACTUALIZAR ESCALA LOCAL ---
    diag_local = np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent())
    print(f"Nueva diagonal local: {diag_local:.3f}")

    # --- 3. ELIMINACIÓN DE SUELO (GUILLOTINA) ---
    if remove_floor:
        thr = diag_local * floor_thr
        print(f"Buscando plano base para corte (grosor {thr:.3f})...")
        model, inliers = pcd.segment_plane(distance_threshold=thr, ransac_n=3, num_iterations=1000)

        if len(inliers) > 0:
            A, B, C, D = model
            pts = np.asarray(pcd.points)
            
            normal = np.array([A, B, C])
            center = np.median(pts, axis=0)
            if np.dot(center, normal) + D < 0:
                A, B, C, D = -A, -B, -C, -D
                
            distances = pts[:, 0]*A + pts[:, 1]*B + pts[:, 2]*C + D
            mask = distances > (thr * 0.5) 
            
            pcd = pcd.select_by_index(np.where(mask)[0])
            print(f"-> Guillotina aplicada. Eliminados {np.sum(~mask):,} puntos.")

    # --- 4. CLUSTERING ESTRICTO ---
    eps_dist = diag_local * cluster_eps
    print(f"Agrupando (DBSCAN) con distancia estricta: {eps_dist:.3f}...")
    
    labels = np.array(pcd.cluster_dbscan(eps=eps_dist, min_points=cluster_min))
    if labels.max() >= 0:
        counts = np.bincount(labels[labels >= 0])
        best_label = counts.argmax()
        print(f"-> Encontradas {labels.max() + 1} islas. Manteniendo la mayor ({counts[best_label]:,} puntos).")
        pcd = pcd.select_by_index(np.where(labels == best_label)[0])
    else:
        print("-> Advertencia: No se formaron grupos consistentes.")

    # --- 5. LIMPIEZA FINA ESTADÍSTICA ---
    print("Limpiando pelusa final flotante...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)

    # --- 6. GUARDADO ---
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Final: {len(pcd.points):,} puntos guardados ({100*len(pcd.points)/n0:.1f}% del original)")
    
    return pcd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--min-opacity", type=float, default=0.3, help="solo 3DGS; 0 = off")
    ap.add_argument("--crop-ratio", type=float, default=0.035, help="Radio de corte. 0 para desactivar.")
    ap.add_argument("--keep-floor", action="store_true", help="Desactivar corte de guillotina para el suelo.")
    ap.add_argument("--floor-thr", type=float, default=0.015, help="Grosor del suelo (ratio vs diagonal local)")
    ap.add_argument("--cluster-eps", type=float, default=0.08, help="Distancia DBSCAN")
    ap.add_argument("--cluster-min", type=int, default=50, help="Mínimo de puntos por isla")
    a = ap.parse_args()

    clean_point_cloud(
        inp_path=a.inp,
        out_path=a.out,
        min_opacity=a.min_opacity,
        crop_ratio=a.crop_ratio,
        floor_thr=a.floor_thr,
        cluster_eps=a.cluster_eps,
        cluster_min=a.cluster_min,
        remove_floor=not a.keep_floor  # <- Aquí está la magia, por defecto es True
    )