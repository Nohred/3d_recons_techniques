"""Reconstrucción por fusión RGBD de las 3 cámaras fijas, con poses estimadas por ChArUco.

Por objeto:
  1. Lee raw/calibration/calibration_inputs.json (K, distorsión, tamaño del tablero).
  2. Estima la pose de cada cámara respecto al tablero (raw/calibration/camera_0i/rgb.png, solvePnP).
  3. Retroproyecta raw/rgbd/camera_0i/depth.npy (z al plano de imagen; 0 = inválido) al marco
     del tablero, fusiona las 3 vistas y recorta suelo/entorno con la geometría conocida.
  4. Escribe reconstruction/exports/rgbd_fusion/{fused.ply, poses.json}.

La verdad del simulador (simulation_private/) solo se usa para reportar el error de pose.
Uso: python3 src/reconstruct/rgbd_fusion.py [objeto ...] [--overwrite]
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, Path(__file__).resolve().parent / "cameras"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ================= CONFIGURACION EDITABLE =================
OBJETOS = None          # None: todos los objetos con raw/rgbd y raw/calibration
SOBRESCRIBIR = False
CAMARAS = ("camera_00", "camera_01", "camera_02")
MIN_ESQUINAS = 6        # esquinas ChArUco mínimas para aceptar la pose de una cámara
MAX_REPROYECCION_PX = 2.0
MARGEN_ENCUADRE = 1.20  # MARGEN de generate_rgbd_three_cameras.py (para estimar el radio del objeto)
FACTOR_RADIO_RECORTE = 1.3
MARGEN_SUELO_M = 0.003  # se descartan puntos a menos de esto sobre el plano del tablero
VOXELES_POR_RADIO = 250  # tamaño de voxel = radio_estimado / VOXELES_POR_RADIO
# ==========================================================


def candidates(objects) -> list[Path]:
    data_root = ROOT / "data"
    if objects is None:
        return sorted(p for p in data_root.iterdir()
                      if (p / "raw" / "rgbd").is_dir()
                      and (p / "raw" / "calibration" / "calibration_inputs.json").is_file())
    return [data_root / o for o in objects]


def record(object_dir: Path, status: str, seconds: int, detail: str) -> None:
    """Mismo formato que record() de src/reconstruct/lib.sh."""
    rec = object_dir / "reconstruction"
    rec.mkdir(parents=True, exist_ok=True)
    line = (f'{datetime.now().astimezone().isoformat(timespec="seconds")},{object_dir.name},'
            f'rgbd_fusion,{status},{seconds},"{detail.replace(chr(34), chr(39))}"\n')
    for f in (rec / "stages.csv", ROOT / "reports" / "reconstruction_summary.csv"):
        f.parent.mkdir(parents=True, exist_ok=True)
        if not f.exists():
            f.write_text("timestamp,object,stage,status,seconds,detail\n")
        with f.open("a") as fh:
            fh.write(line)


def matrix_k(k: dict) -> np.ndarray:
    return np.array([[k["fx"], 0.0, k["cx"]], [0.0, k["fy"], k["cy"]], [0.0, 0.0, 1.0]])


def estimate_pose(image_path: Path, board, K: np.ndarray, dist: np.ndarray) -> dict:
    """T_tablero<-camara (OpenCV) a partir de esquinas ChArUco."""
    import cv2
    from PIL import Image

    gray = cv2.cvtColor(np.asarray(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2GRAY)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    n = 0 if ids is None else len(ids)
    if n < MIN_ESQUINAS:
        raise RuntimeError(f"{image_path}: solo {n} esquinas ChArUco (< {MIN_ESQUINAS})")
    obj_pts, img_pts = board.matchImagePoints(corners, ids)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError(f"{image_path}: solvePnP falló")
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img_pts.reshape(-1, 2)) ** 2, axis=1))))
    if err > MAX_REPROYECCION_PX:
        raise RuntimeError(f"{image_path}: reproyección {err:.2f}px > {MAX_REPROYECCION_PX}px")
    R, _ = cv2.Rodrigues(rvec)
    T_cam_board = np.eye(4)
    T_cam_board[:3, :3], T_cam_board[:3, 3] = R, tvec.ravel()
    return {"T_board_cam": np.linalg.inv(T_cam_board), "corners": n, "reprojection_px": err}


def backproject(rgb_path: Path, depth_path: Path, K: np.ndarray, T_board_cam: np.ndarray):
    from PIL import Image

    depth = np.load(depth_path, allow_pickle=False).astype(np.float64)
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float64) / 255.0
    if depth.shape != rgb.shape[:2]:
        raise RuntimeError(f"Tamaños distintos: {depth_path} {depth.shape} vs {rgb.shape[:2]}")
    v, u = np.nonzero(np.isfinite(depth) & (depth > 0))
    z = depth[v, u]
    pts = np.column_stack([(u - K[0, 2]) * z / K[0, 0],
                           (v - K[1, 2]) * z / K[1, 1], z])
    pts = pts @ T_board_cam[:3, :3].T + T_board_cam[:3, 3]
    return pts, rgb[v, u]


def optical_axes_center(poses: list[np.ndarray]) -> np.ndarray:
    """Punto más cercano (mínimos cuadrados) a los ejes ópticos: centro del objeto encuadrado."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for T in poses:
        c, d = T[:3, 3], T[:3, 2] / np.linalg.norm(T[:3, 2])
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.solve(A, b)


def pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> dict:
    dR = T_est[:3, :3].T @ T_gt[:3, :3]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    return {"rot_deg": ang, "trans_m": float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))}


def fuse_object(object_dir: Path) -> str:
    import open3d as o3d
    import charuco_board as cb

    calib = object_dir / "raw" / "calibration"
    inputs = json.loads((calib / "calibration_inputs.json").read_text(encoding="utf-8"))
    K_cal, K_rgbd = matrix_k(inputs["K_calibracion"]), matrix_k(inputs["K_rgbd"])
    dist = np.asarray(inputs.get("distorsion", [0, 0, 0, 0, 0]), dtype=np.float64)
    tab = inputs["tablero"]
    if (tab["casillas_x"], tab["casillas_y"], tab["diccionario"]) != (cb.CASILLAS_X, cb.CASILLAS_Y, cb.DICCIONARIO):
        raise RuntimeError("El tablero de calibration_inputs.json no coincide con charuco_board.py")
    board, _ = cb.crear_tablero(tab["longitud_casilla_m"])

    cams = []
    for name in CAMARAS:
        pose = estimate_pose(calib / name / "rgb.png", board, K_cal, dist)
        pose["camera"] = name
        cams.append(pose)

    poses = [c["T_board_cam"] for c in cams]
    center = optical_axes_center(poses)
    distance = float(np.mean([np.linalg.norm(T[:3, 3] - center) for T in poses]))
    w, h = inputs["K_rgbd"]["ancho"], inputs["K_rgbd"]["alto"]
    half_fov = min(np.arctan(w / 2 / K_rgbd[0, 0]), np.arctan(h / 2 / K_rgbd[1, 1]))
    radius = distance * np.sin(half_fov) / MARGEN_ENCUADRE

    all_pts, all_rgb = [], []
    for cam in cams:
        rgbd = object_dir / "raw" / "rgbd" / cam["camera"]
        pts, rgb = backproject(rgbd / "rgb.png", rgbd / "depth.npy", K_rgbd, cam["T_board_cam"])
        all_pts.append(pts)
        all_rgb.append(rgb)
    pts, rgb = np.concatenate(all_pts), np.concatenate(all_rgb)

    # Marco del tablero: Z apunta hacia dentro del tablero (hacia el suelo); el objeto queda en z < 0.
    keep = (np.linalg.norm(pts - center, axis=1) < FACTOR_RADIO_RECORTE * radius) & (pts[:, 2] < -MARGEN_SUELO_M)
    if keep.sum() == 0:
        raise RuntimeError("El recorte por geometría eliminó todos los puntos")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[keep])
    pcd.colors = o3d.utility.Vector3dVector(rgb[keep])
    pcd = pcd.voxel_down_sample(radius / VOXELES_POR_RADIO)

    out = object_dir / "reconstruction" / "exports" / "rgbd_fusion"
    out.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out / "fused.ply"), pcd)

    report = {
        "object_id": object_dir.name,
        "frame": "tablero ChArUco (OpenCV): X derecha, Y abajo, Z hacia el suelo; metros",
        "points_total": int(len(pts)), "points_fused": int(len(pcd.points)),
        "center_estimated": center.tolist(), "radius_estimated_m": float(radius),
        "cameras": [{"camera": c["camera"], "corners": c["corners"],
                     "reprojection_px": c["reprojection_px"],
                     "T_board_cam": c["T_board_cam"].tolist()} for c in cams],
    }
    truth_path = object_dir / "simulation_private" / "calibration_ground_truth.json"
    if truth_path.is_file():
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        T_wb_inv = np.linalg.inv(np.asarray(truth["T_mundo_tablero"]))
        for c, T_wc in zip(report["cameras"], truth["T_mundo_camara_opencv"]):
            c["error_vs_simulator"] = pose_error(np.asarray(c["T_board_cam"]), T_wb_inv @ np.asarray(T_wc))
    (out / "poses.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    errs = [c.get("error_vs_simulator", {}).get("trans_m") for c in report["cameras"]]
    errs = [e for e in errs if e is not None]
    return (f"puntos={len(pcd.points)} reproy_max={max(c['reprojection_px'] for c in cams):.2f}px"
            + (f" err_pose_max={max(errs) * 1000:.1f}mm" if errs else ""))


def main() -> None:
    from src.cli_utils import parse_cli

    objects, overwrite = parse_cli(OBJETOS, SOBRESCRIBIR, __doc__)
    failures, done, skipped = [], 0, 0
    for object_dir in candidates(objects):
        fused = object_dir / "reconstruction" / "exports" / "rgbd_fusion" / "fused.ply"
        if fused.is_file() and fused.stat().st_size > 0 and not overwrite:
            skipped += 1
            continue
        print(f"=== {object_dir.name} :: rgbd_fusion ===", flush=True)
        t0 = time.monotonic()
        try:
            detail = fuse_object(object_dir)
            print(f"  {detail}")
            record(object_dir, "ok", round(time.monotonic() - t0), detail)
            done += 1
        except Exception as error:
            traceback.print_exc()
            record(object_dir, "failed", round(time.monotonic() - t0), str(error))
            failures.append(object_dir.name)
    print(f"Fusión RGBD: {done} nuevas, {skipped} existentes omitidas, {len(failures)} fallos.")
    if failures:
        raise SystemExit(f"Fallaron: {failures}")


if __name__ == "__main__":
    main()
