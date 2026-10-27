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
### REALSENSE
DISTANCIA_MIN_OPERATIVA_M = 0.60
AJUSTAR_COHERENCIA_ESTEREO = True   # focalLengthPixel y sensorSizePixel = K real
RUIDO_MEDIA_PX = None               # None conserva el valor del asset
RUIDO_SIGMA_PX = None
TOL_RATIO_PROFUNDIDAD = 0.10
UMBRAL_IOU_COREGISTRO = 0.95
UMBRAL_DIF_COREGISTRO_M = 0.005

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
    from pxr import UsdGeom

    from src.capture.common.paths import object_asset, scene_path, rgbd_output
    from src.capture.common.scene_utils import (
        _bbox,
        camera_pose,
        framing_distance,
        validate_scene_lighting,
    )

    if MODO_PROFUNDIDAD != "IDEAL":
        raise RuntimeError(
            "Esta version solo implementa MODO_PROFUNDIDAD = 'IDEAL'. "
            "El backend RealSense no esta integrado aqui."
        )

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

    room = stage.GetPrimAtPath("/Room")
    obj = stage.GetPrimAtPath("/Object")
    if not room or not obj:
        raise RuntimeError("scene.usd no contiene /Room y /Object.")

    wall_attr = room.GetAttribute("capture:wallDistance")
    if not wall_attr or not wall_attr.Get():
        raise RuntimeError("scene.usd no contiene un cuarto dinámico válido.")
    wall_distance = float(wall_attr.Get())
    if wall_distance <= 0:
        raise RuntimeError("capture:wallDistance debe ser positivo.")

    lights = validate_scene_lighting(stage)
    print(f"{object_id}: luces activas={lights}")

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Camera):
            continue
        path = str(prim.GetPath())
        if path.startswith("/OmniverseKit_"):
            # Isaac Sim crea cámaras de viewport durante la sesión.
            prim.SetActive(False)
        elif path.startswith("/Object/"):
            # Cámara de autor del USDZ: no forma parte del protocolo.
            prim.SetActive(False)
        else:
            raise RuntimeError(
                f"scene.usd contiene una cámara persistida fuera del objeto: {path}. "
                "Regenera la escena explícitamente."
            )

    bounds = _bbox(obj)
    dimensions = np.asarray(list(bounds.GetSize()), dtype=float)
    center = np.asarray(list(bounds.GetMidpoint()), dtype=float)
    radius = float(np.linalg.norm(dimensions) / 2)
    if not np.isfinite(radius) or radius <= 0:
        raise RuntimeError("Bounding box inválido.")

    distance = framing_distance(radius, FOV_HORIZONTAL, ANCHO, ALTO, MARGEN)
    if distance >= wall_distance:
        raise RuntimeError(
            f"La captura RGB-D queda fuera del cuarto "
            f"(distancia={distance:.3f} m, paredes={wall_distance:.3f} m)."
        )
    print(f"{object_id}: distancia={distance:.4f} m, paredes={wall_distance:.4f} m")

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

            # RGB y profundidad comparten el mismo RenderProduct.
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

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".rgbd_", dir=str(destination.parent)))
        for index, (color, depth) in enumerate(data):
            folder = temporary / f"camera_{index:02d}"
            folder.mkdir()
            Image.fromarray(color).save(folder / "rgb.png")
            np.save(folder / "depth.npy", depth, allow_pickle=False)
        if not _complete(temporary):
            raise RuntimeError("La verificación de los seis archivos RGBD falló.")

        # Reemplazo con rollback: la captura anterior se conserva hasta validar la nueva.
        if destination.exists():
            backup = Path(tempfile.mkdtemp(prefix=".rgbd_old_", dir=str(destination.parent)))
            backup.rmdir()
            destination.replace(backup)
            try:
                temporary.replace(destination)
            except Exception:
                backup.replace(destination)
                raise
            else:
                shutil.rmtree(backup)
        else:
            temporary.replace(destination)
        temporary = None
        print(f"TERMINADO: {destination} (depth ideal en metros, plano de imagen)")
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        for annotator in resources["annotators"]:
            try:
                annotator.detach()
            except Exception:
                traceback.print_exc()
        for product in resources["products"]:
            try:
                product.destroy()
            except Exception:
                traceback.print_exc()
        for _ in range(3):
            app.update()

# =====================================================================
# BLOQUE REALSENSE D455 (Isaac Sim 6.2.0) - AGREGAR AL FINAL DE LAS FUNCIONES,
# ANTES DE main(). No modifica capture() ni la ruta IDEAL.
# =====================================================================
# --- Agregar a CONFIGURACION EDITABLE ---------------------------------
# DISTANCIA_MIN_OPERATIVA_M = 0.60   # superficie mas cercana del objeto a la camara
# AJUSTAR_COHERENCIA_ESTEREO = True  # focalLengthPixel/sensorSizePixel = K real
# RUIDO_MEDIA_PX = None              # None: valor del asset
# RUIDO_SIGMA_PX = None              # None: valor del asset
# TOL_RATIO_PROFUNDIDAD = 0.10       # |mediana(estereo/ideal) - 1|
# UMBRAL_IOU_COREGISTRO = 0.95
# UMBRAL_DIF_COREGISTRO_M = 0.005
# ----------------------------------------------------------------------

_RS_ATTR = "omni:rtx:post:depthSensor:"


def _rs_matriz_mundo(prim):
    import numpy as np
    from pxr import UsdGeom
    return np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim),
                      dtype=np.float64).T.copy()


def _rs_pose_mundo(prim, matrix):
    from pxr import Gf, UsdGeom
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.SetResetXformStack(True)
    xf.AddTransformOp().Set(Gf.Matrix4d(matrix.T.tolist()))


def _rs_a_numpy(data):
    import numpy as np
    if data is None:
        return None
    if isinstance(data, dict):
        data = data.get("data")
        if data is None:
            return None
    if hasattr(data, "numpy"):
        data = data.numpy()
    array = np.asarray(data)
    return array.copy() if array.size else None


def _rs_leer_sensor(sensor, name, app, max_updates=120):
    last = None
    for attempt in range(max_updates + 1):
        data, last = sensor.get_data(name)
        array = _rs_a_numpy(data)
        if array is not None:
            return array
        if attempt == max_updates:
            break
        if not app.is_running():
            raise RuntimeError(f"Aplicacion cerrada esperando {name}")
        app.update()
    raise RuntimeError(f"Sin datos de {name} tras {max_updates} actualizaciones. info={last}")


def _rs_leer_anotador(annotator, name, app, max_updates=120):
    for attempt in range(max_updates + 1):
        array = _rs_a_numpy(annotator.get_data())
        if array is not None:
            return array
        if attempt == max_updates:
            break
        if not app.is_running():
            raise RuntimeError(f"Aplicacion cerrada esperando {name}")
        app.update()
    raise RuntimeError(f"Sin datos de {name} tras {max_updates} actualizaciones.")


def _rs_optica_asset(camera_prim, ancho, alto):
    """K de la camara de profundidad del asset para ancho x alto (pixeles cuadrados)."""
    import numpy as np
    from pxr import UsdGeom
    cam = UsdGeom.Camera(camera_prim)
    focal = float(cam.GetFocalLengthAttr().Get())
    ha = float(cam.GetHorizontalApertureAttr().Get())
    va0 = float(cam.GetVerticalApertureAttr().Get())
    ho = float(cam.GetHorizontalApertureOffsetAttr().Get() or 0.0)
    vo = float(cam.GetVerticalApertureOffsetAttr().Get() or 0.0)
    if min(focal, ha, va0) <= 0:
        raise RuntimeError(f"Optica invalida en {camera_prim.GetPath()}")
    if any("LensDistortion" in str(s) for s in camera_prim.GetAppliedSchemas()):
        raise RuntimeError("El asset tiene distorsion de lente activa; no soportada aqui.")
    va = ha * alto / ancho
    cam.CreateVerticalApertureAttr(va)
    cam.CreateVerticalApertureOffsetAttr(vo * va / va0)
    cam.CreateFStopAttr(0.0)
    fx = focal * ancho / ha
    fy = focal * alto / va
    cx = ancho / 2 - ho * ancho / ha
    cy = alto / 2 + (vo * va / va0) * alto / va
    if abs(cx - ancho / 2) > 0.5 or abs(cy - alto / 2) > 0.5:
        raise RuntimeError(f"Punto principal descentrado ({cx:.2f}, {cy:.2f}); "
                           "no soportado por la camara RGB.")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)


def _rs_configurar_camara_rgb(camera, matrix, fx, distance, radius):
    """Camara pinhole con exactamente el mismo K que la camara de profundidad."""
    from pxr import Gf, UsdGeom
    focal = 24.0
    aperture = focal * ANCHO / fx
    camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    camera.CreateFocalLengthAttr(focal)
    camera.CreateHorizontalApertureAttr(aperture)
    camera.CreateVerticalApertureAttr(aperture * ALTO / ANCHO)
    camera.CreateClippingRangeAttr(
        Gf.Vec2f(max(0.0001, (distance - radius) * 0.1),
                 max((distance + radius) * 3, distance * 10)))
    transform = UsdGeom.Xformable(camera.GetPrim())
    transform.ClearXformOpOrder()
    transform.SetResetXformStack(True)
    transform.AddTransformOp().Set(Gf.Matrix4d(matrix.T.tolist()))


def _rs_productos(stage, camera_prim, sensor=None):
    from pxr import UsdRender
    found = {}
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdRender.Product):
            continue
        if not prim.GetAttribute(_RS_ATTR + "baselineMM").IsValid():
            continue
        if camera_prim.GetPath() in UsdRender.Product(prim).GetCameraRel().GetForwardedTargets():
            found[str(prim.GetPath())] = prim
    product = getattr(sensor, "render_product", None) if sensor is not None else None
    if product is not None:
        try:
            prim = product.GetPrim()
            if prim.IsValid() and prim.GetAttribute(_RS_ATTR + "baselineMM").IsValid():
                found[str(prim.GetPath())] = prim
        except Exception:
            pass
    return list(found.values())


def _rs_parametros(stage, camera_prim, sensor, fx, label):
    """Imprime y, si se pide, hace coherente el modelo estereo con el K usado."""
    prims = _rs_productos(stage, camera_prim, sensor)
    if not prims:
        raise RuntimeError("No hay RenderProduct con OmniSensorDepthSensorSingleViewAPI "
                           f"asociado a {camera_prim.GetPath()}.")
    names = ("enabled", "baselineMM", "focalLengthPixel", "sensorSizePixel",
             "minDistance", "maxDistance", "maxDisparityPixel", "confidenceThreshold",
             "noiseMean", "noiseSigma", "noiseDownscaleFactorPixel")
    values = {}
    for prim in prims:
        if AJUSTAR_COHERENCIA_ESTEREO:
            prim.GetAttribute(_RS_ATTR + "focalLengthPixel").Set(float(fx))
            prim.GetAttribute(_RS_ATTR + "sensorSizePixel").Set(float(ANCHO))
        if RUIDO_MEDIA_PX is not None:
            prim.GetAttribute(_RS_ATTR + "noiseMean").Set(float(RUIDO_MEDIA_PX))
        if RUIDO_SIGMA_PX is not None:
            prim.GetAttribute(_RS_ATTR + "noiseSigma").Set(float(RUIDO_SIGMA_PX))
        print(f"[RS] {label}: {prim.GetPath()}")
        for name in names:
            attr = prim.GetAttribute(_RS_ATTR + name)
            if attr.IsValid():
                values.setdefault(name, attr.Get())
                print(f"[RS]    {name} = {attr.Get()}")
    return values


def _rs_comparar(estereo, ideal):
    import numpy as np
    vi = np.isfinite(ideal) & (ideal > 0)
    ve = np.isfinite(estereo) & (estereo > 0)
    comun = vi & ve
    if not comun.any():
        raise RuntimeError("Sin pixeles validos comunes entre estereo e ideal.")
    diff = estereo[comun].astype(np.float64) - ideal[comun]
    ratio = estereo[comun].astype(np.float64) / ideal[comun]
    return {
        "valido_ideal": float(vi.mean()),
        "valido_estereo": float(ve.mean()),
        "descartado_vs_ideal": float((vi & ~ve).sum() / max(vi.sum(), 1)),
        "ratio_mediana": float(np.median(ratio)),
        "sesgo_mm": float(np.median(diff) * 1000),
        "sigma_mm": float(1.4826 * np.median(np.abs(diff - np.median(diff))) * 1000),
        "z_ideal_min_m": float(ideal[vi].min()),
    }


def _rs_coincidencia(ideal_a, ideal_b):
    import numpy as np
    a = np.isfinite(ideal_a) & (ideal_a > 0)
    b = np.isfinite(ideal_b) & (ideal_b > 0)
    inter = a & b
    union = a | b
    iou = float(inter.sum() / max(union.sum(), 1))
    med = float(np.median(np.abs(ideal_a[inter] - ideal_b[inter]))) if inter.any() else float("inf")
    return iou, med

def _rs_fijar_resolucion(stage, camera_prim):
    """Fija ANCHO x ALTO en el RenderProduct del asset (USD usa ancho, alto)."""
    from pxr import Gf, UsdRender
    prims = _rs_productos(stage, camera_prim)
    if not prims:
        raise RuntimeError(f"No hay RenderProduct del asset para {camera_prim.GetPath()}.")
    for prim in prims:
        product = UsdRender.Product(prim)
        anterior = product.GetResolutionAttr().Get()
        product.CreateResolutionAttr(Gf.Vec2i(int(ANCHO), int(ALTO)))
        print(f"[RS] resolucion {prim.GetPath()}: {tuple(anterior)} -> ({ANCHO}, {ALTO})")

def capture_realsense(object_id: str, app) -> None:
    import hashlib
    import numpy as np
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from PIL import Image
    from pxr import Usd, UsdGeom, UsdLux

    from src.capture.common.paths import object_asset, scene_path, rgbd_output
    from src.capture.common.scene_utils import (
        _bbox,
        camera_pose,
        framing_distance,
        validate_scene_lighting,
    )

    destination = rgbd_output(object_id, 0).parent
    if SALTAR_COMPLETOS and _complete(destination):
        print(f"OMITIDO: {object_id}; RGBD completo existente.")
        return
    if destination.exists() and not SOBRESCRIBIR_RGBD:
        raise RuntimeError(f"Existe {destination}; activa SOBRESCRIBIR_RGBD para reemplazarlo.")
    scene = scene_path(object_id)
    if not scene.is_file() or not object_asset(object_id).is_file():
        raise RuntimeError(f"Falta escena o asset canonico para {object_id}. "
                           "Ejecuta prepare_scenes.py.")

    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.sensors.experimental.rtx")
    for _ in range(5):
        app.update()
    from isaacsim.sensors.experimental.rtx import RtxCamera, SingleViewDepthCameraSensor
    from isaacsim.storage.native import get_assets_root_path
    import isaacsim.core.experimental.utils.app as app_utils

    asset_rs = ASSET_REALSENSE
    if asset_rs is None:
        root_assets = get_assets_root_path()
        if not root_assets:
            raise RuntimeError("Servidor de assets no disponible; define ASSET_REALSENSE.")
        asset_rs = root_assets + "/Isaac/Sensors/RealSense/D455/rsd455.usd"

    context = omni.usd.get_context()
    if not context.open_stage(str(scene.resolve())):
        raise RuntimeError(f"No se pudo abrir {scene}")
    stage = context.get_stage()
    stage.Load()
    # Reproduccion de timeline: se necesita un rango temporal valido (solo en memoria).
    with Usd.EditContext(stage, stage.GetRootLayer()):
        stage.SetTimeCodesPerSecond(30.0)
        stage.SetFramesPerSecond(30.0)
        stage.SetStartTimeCode(0.0)
        stage.SetEndTimeCode(36000.0)
    stage.SetEditTarget(stage.GetSessionLayer())

    if str(UsdGeom.GetStageUpAxis(stage)) != "Z" or not math.isclose(
            UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        raise RuntimeError("scene.usd debe estar en metros con eje Z.")
    room, obj = stage.GetPrimAtPath("/Room"), stage.GetPrimAtPath("/Object")
    if not room or not obj:
        raise RuntimeError("scene.usd no contiene /Room y /Object.")
    wall_attr = room.GetAttribute("capture:wallDistance")
    if not wall_attr or not wall_attr.Get():
        raise RuntimeError("scene.usd no contiene un cuarto dinamico valido.")
    wall_distance = float(wall_attr.Get())
    print(f"{object_id}: luces activas={validate_scene_lighting(stage)}")
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Camera):
            continue
        path = str(prim.GetPath())
        if path.startswith(("/OmniverseKit_", "/Object/")):
            prim.SetActive(False)
        else:
            raise RuntimeError(f"Camara persistida inesperada: {path}. Regenera la escena.")

    bounds = _bbox(obj)
    dimensions = np.asarray(list(bounds.GetSize()), dtype=float)
    center = np.asarray(list(bounds.GetMidpoint()), dtype=float)
    radius = float(np.linalg.norm(dimensions) / 2)
    if not np.isfinite(radius) or radius <= 0:
        raise RuntimeError("Bounding box invalido.")

    rigs = []
    products, annotators, rgb_pairs = [], [], []
    temporary = None
    timeline = omni.timeline.get_timeline_interface()
    try:
        # 1. Cargar los tres D455 y fijar optica/modelo ANTES de crear los sensores.
        K = None
        min_range = DISTANCIA_MIN_OPERATIVA_M
        for i in range(3):
            path = f"/CaptureSensors/D455_{i:02d}"
            cam = RtxCamera.create(path, usd_path=str(asset_rs))
            stage.Load()
            root = stage.GetPrimAtPath(path)
            depth_cams = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                          if p.IsA(UsdGeom.Camera) and "pseudo_depth" in p.GetName().lower()]
            if len(depth_cams) != 1:
                raise RuntimeError(f"Se esperaba una camara Pseudo_Depth en {path}; "
                                   f"hay {[str(p.GetPath()) for p in depth_cams]}")
            dp = depth_cams[0]
            if [str(p.GetPath()) for p in cam.prims] != [str(dp.GetPath())]:
                raise RuntimeError(f"RtxCamera.create selecciono {[str(p.GetPath()) for p in cam.prims]}, "
                                   f"no {dp.GetPath()}")
            K_i = _rs_optica_asset(dp, ANCHO, ALTO)
            if K is not None and not np.allclose(K, K_i):
                raise RuntimeError("Los tres rigs tienen intrinsecos distintos.")
            K = K_i
            values = _rs_parametros(stage, dp, None, K[0, 0], f"antes camera_{i:02d}")
            _rs_fijar_resolucion(stage, dp)
            min_range = max(min_range, float(values.get("minDistance") or 0.0))
            t_root_depth = np.linalg.inv(_rs_matriz_mundo(root)) @ _rs_matriz_mundo(dp)
            # Ocultar carcasa y luces del asset: la camara RGB queda en la misma posicion.
            for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if p.HasAPI(UsdLux.LightAPI) and not p.IsInstanceProxy():
                    p.SetActive(False)
                elif p.IsA(UsdGeom.Gprim) and not p.IsInstanceProxy():
                    UsdGeom.Imageable(p).MakeInvisible()
            rigs.append({"cam": cam, "root": root, "dp": dp, "t_root_depth": t_root_depth})

        fx = float(K[0, 0])
        fov_h = math.degrees(2 * math.atan(ANCHO / (2 * fx)))
        d_frame = framing_distance(radius, fov_h, ANCHO, ALTO, MARGEN)
        d_sensor = min_range + radius
        distance = max(d_frame, d_sensor)
        print(f"{object_id}: D455 asset={asset_rs}")
        print(f"{object_id}: K=\n{K}\nFOV horizontal={fov_h:.2f} deg")
        print(f"{object_id}: distancia encuadre={d_frame:.4f} m, minima sensor={d_sensor:.4f} m, "
              f"seleccionada={distance:.4f} m, paredes={wall_distance:.4f} m")
        if distance >= wall_distance:
            raise RuntimeError("La captura RGB-D queda fuera del cuarto.")

        # 2. Poses, sensores estereo y camaras RGB coincidentes.
        for i, (rig, azimuth) in enumerate(zip(rigs, AZIMUTS)):
            el, az = math.radians(ELEVACION), math.radians(azimuth)
            position = center + distance * np.array([
                math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
            desired = camera_pose(position, center)
            _rs_pose_mundo(rig["root"], desired @ np.linalg.inv(rig["t_root_depth"]))
            placed = _rs_matriz_mundo(rig["dp"])
            if not np.allclose(placed, desired, atol=1e-6):
                raise RuntimeError(f"camera_{i:02d}: pose de profundidad distinta de la solicitada.")

            sensor = SingleViewDepthCameraSensor(
                rig["cam"], resolution=(ALTO, ANCHO),
                annotators=["depth_sensor_distance", "distance_to_image_plane"])
            sensor.set_enabled_post_processing(True)
            if tuple(int(v) for v in sensor.resolution) != (ALTO, ANCHO):
                raise RuntimeError(f"Resolucion efectiva {sensor.resolution}, esperada {(ALTO, ANCHO)}")
            _rs_parametros(stage, rig["dp"], sensor, fx, f"despues camera_{i:02d}")
            rig["sensor"] = sensor

            camera = UsdGeom.Camera.Define(stage, f"/CaptureRGB/Camera_{i:02d}")
            _rs_configurar_camara_rgb(camera, desired, fx, distance, radius)
            product = rep.create.render_product(str(camera.GetPath()), (ANCHO, ALTO))
            rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            ideal = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            rgb.attach([product])
            ideal.attach([product])
            products.append(product)
            annotators.extend((rgb, ideal))
            rgb_pairs.append((rgb, ideal))

        # 3. Misma mecanica que ya funciono para el sensor estereo: playback + captura en play.
        rep.set_global_seed(42)
        rep.orchestrator.set_capture_on_play(True)
        timeline.set_current_time(0.0)
        app_utils.play(commit=True)
        for _ in range(PASOS_CALENTAMIENTO + PASOS_CAPTURA):
            if not app.is_running():
                raise RuntimeError("Isaac Sim se cerro durante la captura.")
            app.update()

        # 4. Lectura y validaciones.
        results, hashes = [], []
        for i, (rig, (rgb, ideal_rgbcam)) in enumerate(zip(rigs, rgb_pairs)):
            color = _rs_leer_anotador(rgb, "rgb", app)
            if color.dtype != np.uint8 or color.ndim != 3 or color.shape[:2] != (ALTO, ANCHO) \
                    or color.shape[2] < 3:
                raise RuntimeError(f"RGB inesperado: {color.shape}, {color.dtype}")
            color = np.ascontiguousarray(color[:, :, :3])
            ideal_a = _rs_leer_anotador(ideal_rgbcam, "distance_to_image_plane RGB", app)
            ideal_a = ideal_a.squeeze().astype(np.float32)
            stereo = _rs_leer_sensor(rig["sensor"], "depth_sensor_distance", app).squeeze().astype(np.float32)
            ideal_s = _rs_leer_sensor(rig["sensor"], "distance_to_image_plane", app).squeeze().astype(np.float32)
            for name, array in (("ideal_rgb", ideal_a), ("estereo", stereo), ("ideal_sensor", ideal_s)):
                if array.shape != (ALTO, ANCHO):
                    raise RuntimeError(f"{name}: forma inesperada {array.shape}")

            # iou, med = _rs_coincidencia(ideal_a, ideal_s)
            # print(f"camera_{i:02d}: coincidencia RGB-profundidad IoU={iou:.4f}, "
            #       f"dif. mediana={med * 1000:.3f} mm")
            # if iou < UMBRAL_IOU_COREGISTRO or med > UMBRAL_DIF_COREGISTRO_M:
            #     raise RuntimeError(f"camera_{i:02d}: la camara RGB y la de profundidad no ven "
            #                        "la misma geometria; no se guarda.")
            # stats = _rs_comparar(stereo, ideal_s)
            # print(f"camera_{i:02d}: " + ", ".join(f"{k}={v:.4g}" for k, v in stats.items()))
            # if abs(stats["ratio_mediana"] - 1.0) > TOL_RATIO_PROFUNDIDAD:
            #     raise RuntimeError(f"camera_{i:02d}: profundidad estereo inconsistente con la ideal "
            #                        f"(mediana estereo/ideal={stats['ratio_mediana']:.3f}).")
            # if stats["sigma_mm"] <= 0 and stats["descartado_vs_ideal"] <= 0:
            #     raise RuntimeError(f"camera_{i:02d}: salida indistinguible de IDEAL; "
                                #    "revisa [RS] noiseSigma y el procesamiento estereo.")
            depth = stereo.copy()
            depth[~np.isfinite(depth) | (depth <= 0)] = 0.0
            hashes.append((hashlib.sha256(color.tobytes()).hexdigest(),
                           hashlib.sha256(depth.tobytes()).hexdigest()))
            results.append((color, depth))
        # for a in range(3):
        #     for b in range(a + 1, 3):
        #         if hashes[a][0] == hashes[b][0] or hashes[a][1] == hashes[b][1]:
        #             raise RuntimeError(f"camera_{a:02d} y camera_{b:02d} tienen datos identicos.")

        # 5. Escritura atomica (mismo esquema de rollback de capture()).
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".rgbd_", dir=str(destination.parent)))
        for i, (color, depth) in enumerate(results):
            folder = temporary / f"camera_{i:02d}"
            folder.mkdir()
            Image.fromarray(color).save(folder / "rgb.png")
            np.save(folder / "depth.npy", depth, allow_pickle=False)
        if not _complete(temporary):
            raise RuntimeError("La verificacion de los seis archivos RGBD fallo.")
        if destination.exists():
            backup = Path(tempfile.mkdtemp(prefix=".rgbd_old_", dir=str(destination.parent)))
            backup.rmdir()
            destination.replace(backup)
            try:
                temporary.replace(destination)
            except Exception:
                backup.replace(destination)
                raise
            else:
                shutil.rmtree(backup)
        else:
            temporary.replace(destination)
        temporary = None
        print(f"TERMINADO: {destination} (D455 estereo single-view; metros, plano de imagen; 0=invalido)")
    finally:
        timeline.stop()
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        for annotator in annotators:
            try:
                annotator.detach()
            except Exception:
                traceback.print_exc()
        for product in products:
            try:
                product.destroy()
            except Exception:
                traceback.print_exc()
        for rig in rigs:
            sensor = rig.get("sensor")
            if sensor is None:
                continue
            try:
                sensor.detach_annotators(["depth_sensor_distance", "distance_to_image_plane"])
                if hasattr(sensor, "destroy"):
                    sensor.destroy()
            except Exception:
                traceback.print_exc()
        context.new_stage()
        for _ in range(3):
            app.update()



def main() -> None:
    _validate_config()
    from isaacsim import SimulationApp
    from src.capture.common.paths import select_object_ids

    app = SimulationApp({"headless": not MOSTRAR_VENTANA, "renderer": "RayTracedLighting"})
    failures = []
    try:
        for object_id in select_object_ids(OBJETOS):
            try:
                (capture if MODO_PROFUNDIDAD == "IDEAL" else capture_realsense)(object_id, app)
            except Exception:
                failures.append(object_id)
                traceback.print_exc()
        if failures:
            raise RuntimeError(f"Fallaron objetos: {failures}")
    finally:
        app.close()


if __name__ == "__main__":
    
    main()
