"""Captura tres pares RGBD desde la escena canónica de un objeto."""
from pathlib import Path
import math
import shutil
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ================= CONFIGURACION EDITABLE =================
OBJETOS = None  # None: todos los objetos con USDZ; o ["mug_with_coffee"]
MOSTRAR_VENTANA = False
ANCHO, ALTO = 1280, 720
AZIMUTS = (0.0, 120.0, 240.0)
ELEVACION = 25.0
FOV_HORIZONTAL = 50.0
MARGEN = 1.20
MODO_PROFUNDIDAD = "IDEAL"  # "REALSENSE_D455" exige el backend verificado
ASSET_REALSENSE = None
RT_SUBFRAMES = 8
PASOS_CALENTAMIENTO = 40
PASOS_CAPTURA = 8
SALTAR_COMPLETOS = True
SOBRESCRIBIR_RGBD = False
# ==========================================================


def _validate_config() -> None:
    if len(AZIMUTS) != 3 or not all(math.isfinite(x) for x in AZIMUTS):
        raise ValueError("Se requieren exactamente tres azimuts finitos.")
    if not 0 <= ELEVACION < 89 or not 0 < FOV_HORIZONTAL < 170:
        raise ValueError("Elevación o FOV inválidos.")
    if MARGEN <= 1 or min(RT_SUBFRAMES, PASOS_CALENTAMIENTO, PASOS_CAPTURA) < 1:
        raise ValueError("Margen y pasos deben ser positivos.")
    if MODO_PROFUNDIDAD not in {"REALSENSE_D455", "IDEAL"}:
        raise ValueError("MODO_PROFUNDIDAD debe ser REALSENSE_D455 o IDEAL.")


def _complete(folder: Path) -> bool:
    import numpy as np
    from PIL import Image

    try:
        for index in range(3):
            camera = folder / f"camera_{index:02d}"
            with Image.open(camera / "rgb.png") as image:
                if image.size != (ANCHO, ALTO) or image.mode != "RGB":
                    return False
                image.verify()
            depth = np.load(camera / "depth.npy", allow_pickle=False)
            if depth.shape != (ALTO, ANCHO) or depth.dtype != np.float32:
                return False
        return True
    except (OSError, ValueError, EOFError):
        return False


def _configure_camera(camera, matrix, distance: float, radius: float) -> None:
    from pxr import Gf, UsdGeom

    focal = 24.0
    aperture = 2 * focal * math.tan(math.radians(FOV_HORIZONTAL) / 2)
    camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    camera.CreateFocalLengthAttr(focal)
    camera.CreateHorizontalApertureAttr(aperture)
    camera.CreateVerticalApertureAttr(aperture * ALTO / ANCHO)
    camera.CreateClippingRangeAttr(
        Gf.Vec2f(max(0.0001, (distance - radius) * 0.1),
                 max((distance + radius) * 3, distance * 10))
    )
    transform = UsdGeom.Xformable(camera.GetPrim())
    transform.ClearXformOpOrder()
    transform.SetResetXformStack(True)
    transform.AddTransformOp().Set(Gf.Matrix4d(matrix.T.tolist()))


def _realSense_depth_sensor(asset_path: str | None):
    """Carga únicamente el backend D455; no convierte fallos a profundidad ideal."""
    try:
        from isaacsim.sensors.camera import SingleViewDepthSensorAsset
        from isaacsim.sensors.camera import DepthSensorDistance
    except ImportError as error:
        raise RuntimeError(
            "REALSENSE_D455 requiere SingleViewDepthSensorAsset y "
            "DepthSensorDistance en la instalación activa de Isaac Sim; "
            "no se usará profundidad ideal automáticamente."
        ) from error
    if asset_path is None:
        from isaacsim.storage.native import get_assets_root_path

        root = get_assets_root_path()
        if not root:
            raise RuntimeError("No se encontró el servidor de assets de Isaac Sim.")
        asset_path = f"{root}/Isaac/Sensors/Intel/RealSense/rsd455.usd"
    raise RuntimeError(
        "El backend D455 fue detectado, pero esta versión no tiene una "
        "integración verificada de su salida registrada con la cámara RGB. "
        f"Comprueba el asset {asset_path} y la firma de {DepthSensorDistance}."
    )


def _read_pair(rgb_annotator, depth_annotator):
    import numpy as np

    color = np.asarray(rgb_annotator.get_data())
    if color.dtype != np.uint8 or color.ndim != 3 or color.shape[:2] != (ALTO, ANCHO):
        raise RuntimeError(f"RGB inesperado: shape={color.shape}, dtype={color.dtype}")
    depth = np.asarray(depth_annotator.get_data())
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    if depth.shape != (ALTO, ANCHO):
        raise RuntimeError(f"Profundidad inesperada: shape={depth.shape}")
    depth = depth.astype(np.float32, copy=True)
    depth[~np.isfinite(depth) | (depth <= 0)] = 0.0
    if not np.any(depth > 0):
        raise RuntimeError("La profundidad no contiene píxeles válidos.")
    return color[:, :, :3].copy(), depth


def capture(object_id: str, app) -> None:
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import Usd, UsdGeom

    from src.common.paths import ROOM_USD, object_asset, scene_path, rgbd_output
    from src.common.scene_utils import camera_pose, framing_distance, validate_room_fit

    destination = rgbd_output(object_id, 0).parent
    if SALTAR_COMPLETOS and _complete(destination):
        print(f"OMITIDO: {object_id}; RGBD completo existente.")
        return
    if destination.exists() and not SOBRESCRIBIR_RGBD:
        raise RuntimeError(f"Existe una captura incompleta en {destination}; "
                           "activa SOBRESCRIBIR_RGBD para reemplazarla.")
    scene = scene_path(object_id)
    asset = object_asset(object_id)
    if not scene.is_file() or not asset.is_file():
        raise RuntimeError(f"Falta escena o asset canónico para {object_id}. "
                           "Ejecuta prepare_scenes.py.")

    context = omni.usd.get_context()
    if not context.open_stage(str(scene.resolve())):
        raise RuntimeError(f"No se pudo abrir {scene}")
    stage = context.get_stage()
    if str(UsdGeom.GetStageUpAxis(stage)) != "Z" or not math.isclose(
        UsdGeom.GetStageMetersPerUnit(stage), 1.0
    ):
        raise RuntimeError("scene.usd debe estar en metros con eje Z.")
    if not stage.GetPrimAtPath("/Room") or not stage.GetPrimAtPath("/Object"):
        raise RuntimeError("scene.usd no contiene /Room y /Object.")
    room_asset = stage.GetPrimAtPath("/Room").GetAttribute("capture:roomAsset")
    if not room_asset or Path(str(room_asset.Get())).resolve() != ROOM_USD.resolve():
        raise RuntimeError("scene.usd no referencia al room.usd compartido.")
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            raise RuntimeError("scene.usd contiene cámaras; regenérala explícitamente.")

    from src.common.scene_utils import _bbox
    bounds = _bbox(stage.GetPrimAtPath("/Object"))
    dimensions = np.asarray(list(bounds.GetSize()), dtype=float)
    center = np.asarray(list(bounds.GetMidpoint()), dtype=float)
    radius = float(np.linalg.norm(dimensions) / 2)
    distance = framing_distance(radius, FOV_HORIZONTAL, ANCHO, ALTO, MARGEN)
    from src.common.room import ROOM_DEPTH_M, ROOM_HEIGHT_M, ROOM_WIDTH_M
    validate_room_fit(dimensions, center, distance, ROOM_WIDTH_M, ROOM_DEPTH_M, ROOM_HEIGHT_M)

    if MODO_PROFUNDIDAD == "REALSENSE_D455":
        _realSense_depth_sensor(ASSET_REALSENSE)

    resources = {"products": [], "annotators": []}
    pairs = []
    temporary = None
    try:
        for index, azimuth in enumerate(AZIMUTS):
            elevation = math.radians(ELEVACION)
            angle = math.radians(azimuth)
            position = center + distance * np.array([
                math.cos(elevation) * math.cos(angle),
                math.cos(elevation) * math.sin(angle),
                math.sin(elevation),
            ])
            camera = UsdGeom.Camera.Define(stage, f"/CaptureRGB/Camera_{index:02d}")
            _configure_camera(camera, camera_pose(position, center), distance, radius)
            product = rep.create.render_product(str(camera.GetPath()), (ANCHO, ALTO))
            rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            rgb.attach([product])
            depth.attach([product])
            resources["products"].append(product)
            resources["annotators"].extend((rgb, depth))
            pairs.append((rgb, depth))

        rep.orchestrator.set_capture_on_play(False)
        for _ in range(PASOS_CALENTAMIENTO):
            app.update()
        for _ in range(PASOS_CAPTURA):
            if not app.is_running():
                raise RuntimeError("Isaac Sim se cerró durante la captura.")
            rep.orchestrator.step(rt_subframes=RT_SUBFRAMES, delta_time=0.0,
                                  pause_timeline=True)
        data = [_read_pair(*pair) for pair in pairs]
        temporary = Path(tempfile.mkdtemp(prefix=".rgbd_", dir=str(destination.parent)))
        for index, (color, depth) in enumerate(data):
            folder = temporary / f"camera_{index:02d}"
            folder.mkdir()
            Image.fromarray(color).save(folder / "rgb.png")
            np.save(folder / "depth.npy", depth, allow_pickle=False)
        if not _complete(temporary):
            raise RuntimeError("La verificación de los seis archivos RGBD falló.")
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
        temporary = None
        print(f"TERMINADO: {destination} (depth ideal en metros, plano de imagen)")
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        for annotator in resources["annotators"]:
            annotator.detach()
        for product in resources["products"]:
            product.destroy()
        for _ in range(3):
            app.update()


def main() -> None:
    _validate_config()
    from isaacsim import SimulationApp
    from src.common.paths import select_object_ids

    app = SimulationApp({"headless": not MOSTRAR_VENTANA, "renderer": "RayTracedLighting"})
    failures = []
    try:
        for object_id in select_object_ids(OBJETOS):
            try:
                capture(object_id, app)
            except Exception:
                failures.append(object_id)
                traceback.print_exc()
        if failures:
            raise RuntimeError(f"Fallaron objetos: {failures}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
