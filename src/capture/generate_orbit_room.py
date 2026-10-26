"""Capturador oficial RGB en órbita sobre la escena canónica."""
from pathlib import Path
import json
import math
import shutil
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ================= CONFIGURACION EDITABLE =================
OBJETOS = None  # None: todos los objetos; o ["mug_with_coffee"]
MOSTRAR_VENTANA = False
ANCHO, ALTO = 1280, 720
FPS = 30
DURACION_SEGUNDOS = 8
VUELTAS = 1
ELEVACION_INICIAL = 10.0
ELEVACION_FINAL = 65.0
FOV_HORIZONTAL = 50.0
MARGEN = 1.20
RT_SUBFRAMES = 8
PASOS_POR_FOTOGRAMA = 2
CRF = 18
SALTAR_COMPLETOS = True
SOBRESCRIBIR_VIDEO = False
# ==========================================================


def _validate() -> int:
    total = round(FPS * DURACION_SEGUNDOS)
    if ANCHO <= 0 or ALTO <= 0 or ANCHO % 2 or ALTO % 2:
        raise ValueError("La resolución debe ser positiva y par.")
    if FPS <= 0 or DURACION_SEGUNDOS <= 0 or total < 2:
        raise ValueError("FPS y duración deben producir al menos dos frames.")
    if VUELTAS <= 0 or max(abs(ELEVACION_INICIAL), abs(ELEVACION_FINAL)) >= 89:
        raise ValueError("Trayectoria inválida.")
    if not 0 < FOV_HORIZONTAL < 170 or MARGEN <= 1:
        raise ValueError("FOV o margen inválidos.")
    if min(RT_SUBFRAMES, PASOS_POR_FOTOGRAMA) < 1 or not 0 <= CRF <= 51:
        raise ValueError("Parámetros de render inválidos.")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("La órbita requiere ffmpeg y ffprobe.")
    return total


def _encode(frames: Path, video: Path, total: int) -> None:
    subprocess.run([
        shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "warning", "-y",
        "-framerate", str(FPS), "-i", str(frames / "frame_%06d.png"),
        "-frames:v", str(total), "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", str(CRF), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(video),
    ], check=True)
    probe = subprocess.run([
        shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries",
        "stream=nb_read_frames,width,height", "-of", "json", str(video),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    if int(stream["nb_read_frames"]) != total or (
        int(stream["width"]), int(stream["height"])
    ) != (ANCHO, ALTO):
        raise RuntimeError("El video no coincide con los frames solicitados.")


def capture(object_id: str, app, total: int) -> None:
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import Gf, UsdGeom

    from src.common.paths import ROOM_USD, orbit_output, scene_path
    from src.common.scene_utils import _bbox, camera_pose, framing_distance, validate_room_fit
    from src.common.room import ROOM_DEPTH_M, ROOM_HEIGHT_M, ROOM_WIDTH_M

    video = orbit_output(object_id)
    if video.exists() and SALTAR_COMPLETOS:
        print(f"OMITIDO: {video}")
        return
    if video.exists() and not SOBRESCRIBIR_VIDEO:
        raise RuntimeError(f"Existe {video}; activa SOBRESCRIBIR_VIDEO para reemplazarlo.")
    scene = scene_path(object_id)
    if not scene.is_file():
        raise RuntimeError(f"Falta {scene}; ejecuta prepare_scenes.py.")

    context = omni.usd.get_context()
    if not context.open_stage(str(scene.resolve())):
        raise RuntimeError(f"No se pudo abrir {scene}")
    stage = context.get_stage()
    if str(UsdGeom.GetStageUpAxis(stage)) != "Z":
        raise RuntimeError("scene.usd debe usar eje Z.")
    if any(prim.IsA(UsdGeom.Camera) for prim in stage.Traverse()):
        raise RuntimeError("scene.usd contiene cámaras de captura.")
    room_asset = stage.GetPrimAtPath("/Room").GetAttribute("capture:roomAsset")
    if not room_asset or Path(str(room_asset.Get())).resolve() != ROOM_USD.resolve():
        raise RuntimeError("scene.usd no referencia al room.usd compartido.")
    bounds = _bbox(stage.GetPrimAtPath("/Object"))
    dimensions = np.asarray(list(bounds.GetSize()), dtype=float)
    center = np.asarray(list(bounds.GetMidpoint()), dtype=float)
    radius = float(np.linalg.norm(dimensions) / 2)
    distance = framing_distance(radius, FOV_HORIZONTAL, ANCHO, ALTO, MARGEN)
    validate_room_fit(dimensions, center, distance, ROOM_WIDTH_M, ROOM_DEPTH_M, ROOM_HEIGHT_M)

    video.parent.mkdir(parents=True, exist_ok=True)
    frames = Path(tempfile.mkdtemp(prefix=f".orbit_{object_id}_", dir=str(video.parent)))
    product = None
    rgb = None
    try:
        camera = UsdGeom.Camera.Define(stage, "/CaptureRGB/OrbitCamera")
        focal = 24.0
        aperture = 2 * focal * math.tan(math.radians(FOV_HORIZONTAL) / 2)
        camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
        camera.CreateFocalLengthAttr(focal)
        camera.CreateHorizontalApertureAttr(aperture)
        camera.CreateVerticalApertureAttr(aperture * ALTO / ANCHO)
        camera.CreateClippingRangeAttr(Gf.Vec2f(
            max(0.0001, (distance - radius) * 0.1), max((distance + radius) * 3, distance * 10)
        ))
        product = rep.create.render_product("/CaptureRGB/OrbitCamera", (ANCHO, ALTO))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([product])
        rep.orchestrator.set_capture_on_play(False)
        for _ in range(20):
            app.update()
        transform = UsdGeom.Xformable(camera.GetPrim())
        transform.ClearXformOpOrder()
        pose = transform.AddTransformOp()
        for index in range(total):
            u = index / total
            elevation = math.radians(ELEVACION_INICIAL + (ELEVACION_FINAL - ELEVACION_INICIAL) * u)
            azimuth = 2 * math.pi * VUELTAS * u
            position = center + distance * np.array([
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ])
            pose.Set(Gf.Matrix4d(camera_pose(position, center).T.tolist()))
            for _ in range(PASOS_POR_FOTOGRAMA):
                rep.orchestrator.step(rt_subframes=RT_SUBFRAMES, delta_time=0.0,
                                      pause_timeline=True)
            data = np.asarray(rgb.get_data())
            if data.dtype != np.uint8 or data.shape[:2] != (ALTO, ANCHO):
                raise RuntimeError(f"RGB inesperado: {data.shape}, {data.dtype}")
            Image.fromarray(data[:, :, :3].copy()).save(frames / f"frame_{index:06d}.png")
        _encode(frames, video, total)
        print(f"TERMINADO: {video}")
    finally:
        if rgb is not None:
            rgb.detach()
        if product is not None:
            product.destroy()
        shutil.rmtree(frames, ignore_errors=True)
        for _ in range(3):
            app.update()


def main() -> None:
    total = _validate()
    from isaacsim import SimulationApp
    from src.common.paths import select_object_ids

    app = SimulationApp({"headless": not MOSTRAR_VENTANA, "renderer": "RayTracedLighting"})
    failures = []
    try:
        for object_id in select_object_ids(OBJETOS):
            try:
                capture(object_id, app, total)
            except Exception:
                failures.append(object_id)
                traceback.print_exc()
        if failures:
            raise RuntimeError(f"Fallaron objetos: {failures}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
