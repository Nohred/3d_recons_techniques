import argparse
import numpy as np
import open3d as o3d
from plyfile import PlyData

SH_C0 = 0.28209479177387814


def load_pcd(path, min_opacity=0.0):
    """Carga .ply de 3DGS (f_dc/opacity) o de NeRF/otros (red/green/blue)."""
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
    min_opacity=0.0,
    crop_ratio=0.035,
    crop_iters=1,
    floor_thr=0.015,
    cluster_eps=0.08,
    cluster_min=50,
    remove_floor=True,
    std_ratio=1.5,
    center=None,
):
    # --- 1. CARGA ---
    pcd = load_pcd(inp_path, min_opacity)
    n0 = len(pcd.points)
    diag_orig = np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent())
    print(f"Original: {n0:,} puntos | Diagonal: {diag_orig:.3f}")

    # --- 2. RECORTE ESFÉRICO ALREDEDOR DE LA MEDIANA ---
    if crop_ratio > 0:
        limit = diag_orig * crop_ratio
        print(f"Recorte esférico (radio: {limit:.3f}, iteraciones: {crop_iters})...")
        for i in range(crop_iters):
            pts = np.asarray(pcd.points)
            c = np.asarray(center, dtype=np.float64) if (center is not None and i == 0) \
                else np.median(pts, axis=0)
            mask = np.linalg.norm(pts - c, axis=1) < limit
            pcd = pcd.select_by_index(np.where(mask)[0])
            print(f"-> iter {i + 1}: centro {np.round(c, 4)} | quedan {len(pcd.points):,}")
            if len(pcd.points) == 0:
                raise SystemExit("El recorte eliminó todo: sube --crop-ratio o usa --center")

    diag_local = np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent())
    print(f"Nueva diagonal local: {diag_local:.3f}")

    # --- 3. SUELO (GUILLOTINA) ---
    if remove_floor:
        thr = diag_local * floor_thr
        print(f"Buscando plano base (grosor {thr:.3f})...")
        model, inliers = pcd.segment_plane(distance_threshold=thr, ransac_n=3,
                                           num_iterations=1000)
        if len(inliers) > 0:
            A, B, C, D = model
            pts = np.asarray(pcd.points)
            normal = np.array([A, B, C])
            if np.dot(np.median(pts, axis=0), normal) + D < 0:
                A, B, C, D = -A, -B, -C, -D
            dist = pts[:, 0] * A + pts[:, 1] * B + pts[:, 2] * C + D
            mask = dist > (thr * 0.5)
            pcd = pcd.select_by_index(np.where(mask)[0])
            print(f"-> Guillotina aplicada. Eliminados {np.sum(~mask):,} puntos.")

    # --- 4. CLUSTERING ---
    eps_dist = diag_local * cluster_eps
    print(f"DBSCAN (eps {eps_dist:.3f}, min {cluster_min})...")
    labels = np.array(pcd.cluster_dbscan(eps=eps_dist, min_points=cluster_min))
    if len(labels) and labels.max() >= 0:
        counts = np.bincount(labels[labels >= 0])
        best = counts.argmax()
        print(f"-> {labels.max() + 1} islas. Se conserva la mayor ({counts[best]:,}).")
        pcd = pcd.select_by_index(np.where(labels == best)[0])
    else:
        print("-> Advertencia: DBSCAN no formó grupos.")

    # --- 5. FILTRO ESTADÍSTICO ---
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=std_ratio)

    # --- 6. GUARDADO ---
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Final: {len(pcd.points):,} puntos ({100 * len(pcd.points) / n0:.1f}% del original)")
    return pcd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", help="PLY de entrada (3DGS, NeRF, ...)")
    ap.add_argument("out", nargs="?", help="PLY de salida (default: <inp>_clean.ply)")
    ap.add_argument("--min-opacity", type=float, default=0.0, help="solo 3DGS; 0 = off")
    ap.add_argument("--crop-ratio", type=float, default=0.035, help="0 para desactivar")
    ap.add_argument("--crop-iters", type=int, default=1, help="repite recorte re-centrando en la mediana")
    ap.add_argument("--center", type=float, nargs=3, default=None, help="centro manual del 1er recorte")
    ap.add_argument("--keep-floor", action="store_true")
    ap.add_argument("--floor-thr", type=float, default=0.015)
    ap.add_argument("--cluster-eps", type=float, default=0.08)
    ap.add_argument("--cluster-min", type=int, default=50)
    ap.add_argument("--std-ratio", type=float, default=1.5)
    a = ap.parse_args()

    out = a.out or a.inp.rsplit(".", 1)[0] + "_clean.ply"
    clean_point_cloud(
        a.inp, out,
        min_opacity=a.min_opacity, crop_ratio=a.crop_ratio, crop_iters=a.crop_iters,
        center=a.center, floor_thr=a.floor_thr, cluster_eps=a.cluster_eps,
        cluster_min=a.cluster_min, remove_floor=not a.keep_floor, std_ratio=a.std_ratio,
    )