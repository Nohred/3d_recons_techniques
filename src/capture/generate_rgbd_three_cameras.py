"""Isaac Sim 6.2.0: tres rigs D455, profundidad estereo o ideal.
Guardar como src/capture/generate_rgbd_three_cameras.py.
Requiere las escenas y los modulos comunes del repositorio existente.
Salida: rgb.png y depth.npy registrado al RGB (metros, Z optico, 0 invalido).
No exporta sensores, poses, calibraciones, previews ni profundidad ideal.
"""
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
OBJETOS = None  # Para primera prueba: ["gamer_mouse"]
MOSTRAR_VENTANA = False
MODO_PROFUNDIDAD = "REALSENSE_D455"  # Alternativa explicita: "IDEAL" "REALSENSE_D455" 
ASSET_REALSENSE = None  # None: asset oficial de Isaac Sim 6.2.
RGB_RES = (1280, 720)    # (ancho, alto), perfil unico.
DEPTH_RES = (1280, 720)  # Cuadricula nativa antes del registro al RGB.
AZIMUTS = (0.0, 120.0, 240.0)
ELEVACION = 25.0
MARGEN = 1.20
RT_SUBFRAMES = 8
PASOS_CALENTAMIENTO = 60
PASOS_CAPTURA = 12
SALTAR_COMPLETOS = False
SOBRESCRIBIR_RGBD = True  # Activar conscientemente para reemplazar IDEAL previo.
# ==========================================================


def validar():
    if MODO_PROFUNDIDAD not in {"REALSENSE_D455", "IDEAL"}:
        raise ValueError("MODO_PROFUNDIDAD invalido")
    if len(AZIMUTS) != 3 or not all(math.isfinite(a) for a in AZIMUTS):
        raise ValueError("Se necesitan tres azimuts finitos")
    if not 0 <= ELEVACION < 89 or MARGEN <= 1:
        raise ValueError("Elevacion o margen invalidos")
    if any(len(r) != 2 or min(r) <= 0 for r in (RGB_RES, DEPTH_RES)):
        raise ValueError("Resoluciones invalidas")
    if min(RT_SUBFRAMES, PASOS_CALENTAMIENTO, PASOS_CAPTURA) < 1:
        raise ValueError("Pasos invalidos")


def completo(folder):
    import numpy as np
    from PIL import Image
    try:
        for i in range(3):
            c = folder / f"camera_{i:02d}"
            with Image.open(c / "rgb.png") as im:
                if im.size != RGB_RES or im.mode != "RGB":
                    return False
                im.verify()
            d = np.load(c / "depth.npy", allow_pickle=False)
            if d.shape != RGB_RES[::-1] or d.dtype != np.float32:
                return False
            if not np.isfinite(d).all() or not np.any(d > 0):
                return False
        return True
    except (OSError, ValueError, EOFError):
        return False


def matriz_mundo(prim):
    import numpy as np
    from pxr import UsdGeom
    return np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim),
                      dtype=np.float64).T.copy()


def pose_mundo(prim, matrix):
    from pxr import Gf, UsdGeom
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.SetResetXformStack(True)
    xf.AddTransformOp().Set(Gf.Matrix4d(matrix.T.tolist()))


def optica(camera_prim, resolution):
    """Mantiene focal/FOV horizontal del asset; adapta apertura vertical al perfil.
    Pinhole rectificado, pixeles cuadrados; no inventa valores fx/fy de otra D455.
    """
    import numpy as np
    from pxr import UsdGeom
    cam = UsdGeom.Camera(camera_prim)
    w, h = resolution
    focal = float(cam.GetFocalLengthAttr().Get())
    ha = float(cam.GetHorizontalApertureAttr().Get())
    va0 = float(cam.GetVerticalApertureAttr().Get())
    ho = float(cam.GetHorizontalApertureOffsetAttr().Get() or 0)
    vo = float(cam.GetVerticalApertureOffsetAttr().Get() or 0)
    if min(focal, ha, va0) <= 0:
        raise RuntimeError(f"Optica invalida en {camera_prim.GetPath()}")
    # Este perfil es rectificado. No reutilizar distorsiones de otro perfil.
    schemas = [str(s) for s in camera_prim.GetAppliedSchemas()]
    if any("LensDistortion" in s for s in schemas):
        raise RuntimeError("El asset tiene distorsion activa; este capturador espera "
                           "la optica pinhole del asset oficial. No se ignorara.")
    va = ha * h / w
    cam.CreateVerticalApertureAttr(va)
    cam.CreateVerticalApertureOffsetAttr(vo * va / va0)
    cam.CreateFStopAttr(0.0)
    fx, fy = focal * w / ha, focal * h / va
    cx = w / 2 - ho * w / ha
    cy = h / 2 + (vo * va / va0) * h / va
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    # Angulo conservador en el lado mas estrecho de un principal point descentrado.
    half = min(math.atan(cx / fx), math.atan((w-cx) / fx),
               math.atan(cy / fy), math.atan((h-cy) / fy))
    if half <= 0:
        raise RuntimeError("Principal point fuera de la imagen")
    return K, half


def sensor_numpy(sensor, name, app, max_updates=120):
    import numpy as np
    import omni.timeline
    import omni.usd
    import omni.replicator.core as rep
    from pxr import UsdRender

    def convertir(data):
        if data is None:
            return None

        # Los anotadores directos pueden devolver un diccionario.
        if isinstance(data, dict):
            data = data.get("data")
            if data is None:
                return None

        if hasattr(data, "numpy"):
            data = data.numpy()

        array = np.asarray(data)

        if array.size == 0:
            return None

        return array.copy()

    # Mantener el backend de profundidad sin sustituirlo por IDEAL.
    if name != "rgb":
        last_info = None

        for attempt in range(max_updates + 1):
            data, last_info = sensor.get_data(name)
            array = convertir(data)

            if array is not None:
                return array

            if attempt == max_updates:
                break

            if not app.is_running():
                raise RuntimeError(
                    f"Aplicacion cerrada esperando {name}"
                )

            app.update()

        raise RuntimeError(
            f"Sin datos de {name}. "
            f"Anotadores={sensor.annotators}; "
            f"info={last_info}"
        )

    # Primero comprobar el acceso habitual.
    data, info = sensor.get_data("rgb")
    array = convertir(data)

    if array is not None:
        return array

    # Diagnostico y lectura directa del MISMO RenderProduct.
    product = sensor.render_product

    if not product or not product.GetPrim().IsValid():
        raise RuntimeError(
            "CameraSensor no tiene un RenderProduct valido"
        )

    product_path = str(product.GetPath())
    product_prim = product.GetPrim()

    if not product_prim.IsActive():
        raise RuntimeError(
            f"RenderProduct inactivo: {product_path}"
        )

    camera_targets = product.GetCameraRel().GetForwardedTargets()

    print(f"[RGB] RenderProduct: {product_path}")
    print(f"[RGB] Camaras referenciadas: {camera_targets}")
    print(f"[RGB] Anotadores del wrapper: {sensor.annotators}")
    print(
        "[RGB] Resolucion USD:",
        product.GetResolutionAttr().Get(),
    )

    if len(camera_targets) != 1:
        raise RuntimeError(
            f"Se esperaba una camara para {product_path}; "
            f"targets={camera_targets}"
        )

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(camera_targets[0])

    if not camera_prim or not camera_prim.IsActive():
        raise RuntimeError(
            f"Camara RGB inexistente o inactiva: "
            f"{camera_targets[0]}"
        )

    print(
        "[RGB] Tick rate:",
        camera_prim.GetAttribute(
            "omni:sensor:tickRate"
        ).Get(),
    )
    print(
        "[RGB] Schemas:",
        camera_prim.GetAppliedSchemas(),
    )

    timeline = omni.timeline.get_timeline_interface()

    if not timeline.is_playing():
        raise RuntimeError(
            "La timeline esta detenida al intentar leer RGB"
        )

    initial_time = timeline.get_current_time()

    direct_rgb = rep.AnnotatorRegistry.get_annotator(
        "rgb",
        device="cpu",
    )

    try:
        direct_rgb.attach([product_path])

        for attempt in range(max_updates):
            if not app.is_running():
                raise RuntimeError(
                    "Aplicacion cerrada durante diagnostico RGB"
                )

            app.update()

            # Comprobar ambas rutas tras evaluar el render.
            wrapped_data, info = sensor.get_data("rgb")
            wrapped_array = convertir(wrapped_data)

            if wrapped_array is not None:
                print(
                    "[RGB] CameraSensor produjo datos "
                    f"tras {attempt + 1} actualizaciones"
                )
                return wrapped_array

            direct_array = convertir(direct_rgb.get_data())

            if direct_array is not None:
                print(
                    "[RGB] Lectura directa correcta; "
                    "CameraSensor sigue sin devolver datos. "
                    f"shape={direct_array.shape}"
                )
                return direct_array

        final_time = timeline.get_current_time()

        ordered_vars = [
            str(path)
            for path in product.GetOrderedVarsRel()
            .GetForwardedTargets()
        ]

        raise RuntimeError(
            "RGB vacio tanto en CameraSensor como en el "
            "anotador directo del mismo RenderProduct.\n"
            f"RenderProduct={product_path}\n"
            f"Camara={camera_targets[0]}\n"
            f"Timeline: {initial_time:.6f} -> "
            f"{final_time:.6f} segundos\n"
            f"RenderVars={ordered_vars}\n"
            f"Ultima info del wrapper={info}"
        )

    finally:
        direct_rgb.detach()


def registrar(depth, kd, kc, rgb_from_depth):
    """Registro forward con z-buffer; no rellena huecos ni suaviza ruido."""
    import numpy as np
    w, h = RGB_RES
    v, u = np.nonzero(np.isfinite(depth) & (depth > 0))
    z = depth[v, u].astype(np.float64)
    p = np.stack(((u-kd[0, 2])*z/kd[0, 0],
                  (v-kd[1, 2])*z/kd[1, 1], z), axis=0)
    p = rgb_from_depth[:3, :3] @ p + rgb_from_depth[:3, 3:4]
    good = np.isfinite(p).all(axis=0) & (p[2] > 0)
    p = p[:, good]
    x = np.rint(kc[0, 0]*p[0]/p[2] + kc[0, 2]).astype(np.int64)
    y = np.rint(kc[1, 1]*p[1]/p[2] + kc[1, 2]).astype(np.int64)
    good = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out = np.full(w*h, np.inf, dtype=np.float32)
    np.minimum.at(out, y[good]*w+x[good], p[2, good].astype(np.float32))
    out[~np.isfinite(out)] = 0
    return out.reshape(h, w)

def configurar_render_products(stage, camera_prim, resolution):
    """Configura templates existentes antes de crear CameraSensor.

    resolution: (ancho, alto).
    No borra templates ni sus parametros de profundidad.
    """
    from pxr import Gf, UsdRender

    camera_path = camera_prim.GetPath()
    width, height = resolution
    encontrados = []

    for prim in stage.TraverseAll():
        if not prim.IsA(UsdRender.Product):
            continue

        product = UsdRender.Product(prim)
        targets = product.GetCameraRel().GetForwardedTargets()

        if camera_path not in targets:
            continue

        anterior = product.GetResolutionAttr().Get()
        product.CreateResolutionAttr(
            Gf.Vec2i(int(width), int(height))
        )

        encontrados.append(str(prim.GetPath()))
        print(
            f"RenderProduct {prim.GetPath()}: "
            f"{anterior} -> ({width}, {height})"
        )

    return encontrados

def cargar_rig(stage, index, asset):
    import numpy as np
    from pxr import Usd, UsdGeom, UsdLux, Sdf
    from isaacsim.sensors.experimental.rtx import (
        RtxCamera, CameraSensor, SingleViewDepthCameraSensor)
    path = f"/CaptureSensors/D455_{index:02d}"
    depth_camera = RtxCamera.create(path, usd_path=asset)
    stage.Load()
    root = stage.GetPrimAtPath(path)
    cameras = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
               if p.IsA(UsdGeom.Camera)]
    depth = [p for p in cameras if "pseudo_depth" in p.GetName().lower()]
    color = [p for p in cameras if any(s in p.GetName().lower()
                                      for s in ("color", "rgb"))]
    if len(depth) != 1 or len(color) != 1:
        raise RuntimeError(f"Asset D455 inesperado; camaras: "
                           f"{[str(p.GetPath()) for p in cameras]}")
    dp, cp = depth[0], color[0]


    # Autotrigger: renderizar ambos sensores en cada frame.
    for camera_prim in (dp, cp):
        tick = camera_prim.GetAttribute("omni:sensor:tickRate")

        if not tick:
            tick = camera_prim.CreateAttribute(
                "omni:sensor:tickRate",
                Sdf.ValueTypeNames.Float,
                custom=False,
            )

        tick.Set(0.0)

    depth_products = configurar_render_products(
    stage, dp, DEPTH_RES
    )

    configurar_render_products(
        stage, cp, RGB_RES
    )

    if not depth_products:
        raise RuntimeError(
            "No se encontro el RenderProduct de profundidad del D455. "
            "No se creara un sensor sin verificar su template."
        )
    # Confirmar que el helper selecciono la camara de profundidad, no otra.
    selected = [str(p.GetPath()) for p in depth_camera.prims]
    if selected != [str(dp.GetPath())]:
        raise RuntimeError(f"RtxCamera.create selecciono {selected}; "
                           f"se esperaba {dp.GetPath()}. No se usara un sensor incorrecto.")
    kd, hd = optica(dp, DEPTH_RES)
    kc, hc = optica(cp, RGB_RES)
    t_root_depth = np.linalg.inv(matriz_mundo(root)) @ matriz_mundo(dp)
    t_color_depth_usd = np.linalg.inv(matriz_mundo(cp)) @ matriz_mundo(dp)
    # USD: X derecha, Y arriba, -Z frente. Optico: X derecha, Y abajo, +Z frente.
    c = np.diag([1., -1., -1., 1.])
    t_color_depth = c @ t_color_depth_usd @ c
    # Ocultar carcasas, sin ocultar ni mover las camaras ni su rig.
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdLux.LightAPI) and not p.IsInstanceProxy():
            p.SetActive(False)
        elif p.IsA(UsdGeom.Gprim) and not p.IsInstanceProxy():
            UsdGeom.Imageable(p).MakeInvisible()
    real = MODO_PROFUNDIDAD == "REALSENSE_D455"
    names = ["depth_sensor_distance", "distance_to_image_plane"] if real else ["distance_to_image_plane"]
    ds = SingleViewDepthCameraSensor(depth_camera, resolution=DEPTH_RES[::-1],
                                    annotators=names)
    ds.set_enabled_post_processing(real)
    cs = CameraSensor(RtxCamera(str(cp.GetPath()), reset_xform_op_properties=False),
                      resolution=RGB_RES[::-1], annotators=["rgb"])

    expected_depth = DEPTH_RES[::-1]  # (alto, ancho)
    expected_rgb = RGB_RES[::-1]

    actual_depth = tuple(int(v) for v in ds.resolution)
    actual_rgb = tuple(int(v) for v in cs.resolution)

    if actual_depth != expected_depth:
        raise RuntimeError(
            f"Depth: resolucion efectiva {actual_depth}, "
            f"esperada {expected_depth}"
        )

    if actual_rgb != expected_rgb:
        raise RuntimeError(
            f"RGB: resolucion efectiva {actual_rgb}, "
            f"esperada {expected_rgb}"
        )

    print(
        f"Resoluciones verificadas: "
        f"RGB={actual_rgb}, depth={actual_depth} (alto, ancho)"
    )
        
    return dict(root=root, dp=dp, cp=cp, kd=kd, kc=kc,
                root_depth=t_root_depth, color_depth=t_color_depth,
                half=min(hd, hc), ds=ds, cs=cs, names=names)


def verificar_encuadre(rig, bounds):
    import numpy as np
    from itertools import product
    lo, hi = bounds.GetMin(), bounds.GetMax()
    points = np.array([[lo[j] if b[j] == 0 else hi[j] for j in range(3)]
                       for b in product((0, 1), repeat=3)])
    for p, k, res in ((rig["dp"], rig["kd"], DEPTH_RES),
                       (rig["cp"], rig["kc"], RGB_RES)):
        local = (np.linalg.inv(matriz_mundo(p)) @
                 np.column_stack((points, np.ones(8))).T).T
        z = -local[:, 2]
        if np.any(z <= 0):
            raise RuntimeError("Objeto detras de una camara")
        u = k[0, 0]*local[:, 0]/z+k[0, 2]
        v = -k[1, 1]*local[:, 1]/z+k[1, 2]
        if np.any((u < 0) | (u >= res[0]) | (v < 0) | (v >= res[1])):
            raise RuntimeError("El bounding box no cabe en ambas camaras; "
                               "revisar asset o aumentar MARGEN")


def capture(object_id, app):
    import numpy as np
    import omni.usd
    import omni.timeline
    import omni.replicator.core as rep
    from PIL import Image
    from pxr import Usd, UsdGeom
    from isaacsim.storage.native import get_assets_root_path
    from src.common.paths import scene_path, object_asset, rgbd_output
    from src.common.scene_utils import _bbox, camera_pose, validate_scene_lighting
    destination = rgbd_output(object_id, 0).parent
    if destination.exists():
        if SALTAR_COMPLETOS and completo(destination):
            print(f"OMITIDO: {object_id}; no se puede inferir el backend de archivos previos")
            return
        if not SOBRESCRIBIR_RGBD:
            raise RuntimeError(f"Existe {destination}. Para reemplazarlo configura "
                               "SALTAR_COMPLETOS=False y SOBRESCRIBIR_RGBD=True")
    scene = scene_path(object_id)
    if not scene.is_file() or not object_asset(object_id).is_file():
        raise RuntimeError("Falta scene.usd u original.usdz; ejecutar prepare_scenes.py")
    context = omni.usd.get_context()
    if not context.open_stage(str(scene.resolve())):
        raise RuntimeError(f"No se pudo abrir {scene}")
    stage = context.get_stage()
    stage.Load()

    # Metadatos temporales en memoria. No llamar a Save().
    with Usd.EditContext(stage, stage.GetRootLayer()):
        stage.SetTimeCodesPerSecond(30.0)
        stage.SetFramesPerSecond(30.0)
        stage.SetStartTimeCode(0.0)
        stage.SetEndTimeCode(36000.0)

    # Todas las modificaciones de captura se quedan en memoria.
    stage.SetEditTarget(stage.GetSessionLayer())
    if str(UsdGeom.GetStageUpAxis(stage)) != "Z" or not math.isclose(
            UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        raise RuntimeError("Escena requerida: metros, Z vertical")
    room, obj = stage.GetPrimAtPath("/Room"), stage.GetPrimAtPath("/Object")
    if not room or not obj:
        raise RuntimeError("Falta /Room o /Object")
    wall_distance = float(room.GetAttribute("capture:wallDistance").Get())
    if wall_distance <= 0:
        raise RuntimeError("La escena no contiene un cuarto dinámico válido")
    print(f"Luces activas: {validate_scene_lighting(stage)}")
    for p in list(stage.Traverse()):
        if p.IsA(UsdGeom.Camera):
            path = str(p.GetPath())
            if path.startswith(("/Object/", "/OmniverseKit_", "/Capture")):
                p.SetActive(False)
            else:
                raise RuntimeError(f"Camara persistida inesperada: {path}")
    bounds = _bbox(obj)
    dimensions = np.array(list(bounds.GetSize()), dtype=float)
    center = np.array(list(bounds.GetMidpoint()), dtype=float)
    radius = float(np.linalg.norm(dimensions)/2)
    if not np.isfinite(radius) or radius <= 0:
        raise RuntimeError("Bounding box invalido")
    asset = ASSET_REALSENSE
    if asset is None:
        root = get_assets_root_path()
        if not root:
            raise RuntimeError("Servidor de assets no disponible; usar ASSET_REALSENSE local")
        asset = root + "/Isaac/Sensors/RealSense/D455/rsd455.usd"
    rigs, temporary = [], None
    timeline = omni.timeline.get_timeline_interface()
    try:
        for i in range(3):
            rigs.append(cargar_rig(stage, i, str(asset)))
        half = min(r["half"] for r in rigs)
        offset = max(np.linalg.norm(r["color_depth"][:3, 3]) for r in rigs)
        distance = MARGEN*radius/math.sin(half)+offset
        if distance + offset >= wall_distance:
            raise RuntimeError("La trayectoria RGB-D queda fuera del cuarto.")
        for rig, azimuth in zip(rigs, AZIMUTS):
            az, el = math.radians(azimuth), math.radians(ELEVACION)
            position = center + distance*np.array([
                math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
            desired_depth = camera_pose(position, center)
            pose_mundo(rig["root"], desired_depth @ np.linalg.inv(rig["root_depth"]))
            verificar_encuadre(rig, bounds)
            # Verificar que mover el rig no altero las extrinsecas internas.
            c = np.diag([1., -1., -1., 1.])
            measured = c @ np.linalg.inv(matriz_mundo(rig["cp"])) @ matriz_mundo(rig["dp"]) @ c
            if not np.allclose(measured, rig["color_depth"], atol=1e-6):
                raise RuntimeError("Se altero la extrinseca RGB-depth del asset")
        print(f"{object_id}: D455, modo={MODO_PROFUNDIDAD}, distancia={distance:.4f} m")
        print("K profundidad nativa:\n", rigs[0]["kd"])
        print("K RGB y depth.npy registrado:\n", rigs[0]["kc"])
        print("T_RGB_desde_DEPTH (optico, metros):\n", rigs[0]["color_depth"])
        rep.set_global_seed(42)

        # Este capturador avanza mediante app.update():
        # permitir que Replicator capture durante playback.
        rep.orchestrator.set_capture_on_play(True)

        

        import isaacsim.core.experimental.utils.app as app_utils

        timeline.set_current_time(0.0)
        app_utils.play(commit=True)

        for _ in range(PASOS_CALENTAMIENTO + PASOS_CAPTURA):
            if not app.is_running():
                raise RuntimeError("Aplicacion cerrada durante calentamiento")

            app.update()

            
        results = []
        for i, rig in enumerate(rigs):
            rgb = sensor_numpy(rig["cs"], "rgb", app)
            name = "depth_sensor_distance" if MODO_PROFUNDIDAD == "REALSENSE_D455" else "distance_to_image_plane"
            depth = sensor_numpy(rig["ds"], name, app).squeeze().astype(np.float32)
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[:2] != RGB_RES[::-1] or rgb.shape[2] < 3:
                raise RuntimeError(f"RGB inesperado: {rgb.shape}, {rgb.dtype}")
            if depth.shape != DEPTH_RES[::-1]:
                raise RuntimeError(f"Depth inesperado: {depth.shape}")
            depth[~np.isfinite(depth) | (depth <= 0)] = 0
            if not np.any(depth > 0):
                raise RuntimeError("Profundidad simulada sin mediciones validas")
            if MODO_PROFUNDIDAD == "REALSENSE_D455":
                ideal = sensor_numpy(rig["ds"], "distance_to_image_plane", app).squeeze()
                ok = (depth > 0) & np.isfinite(ideal) & (ideal > 0)
                if not np.any(ok):
                    raise RuntimeError("No hay pixeles comunes para verificar el backend")
                diff = np.abs(depth[ok]-ideal[ok])
                print(f"camera_{i:02d}: diferencia vs ideal: "
                      f"mediana={np.median(diff):.6g} m, max={diff.max():.6g} m")
                if np.all(diff <= 1e-7) and np.array_equal(depth > 0, np.isfinite(ideal) & (ideal > 0)):
                    raise RuntimeError("Salida indistinguible de IDEAL: revisar parametros "
                                       "de ruido y plantilla RenderProduct del asset. "
                                       "No se guardara como RealSense validada.")
            aligned = registrar(depth, rig["kd"], rig["kc"], rig["color_depth"])
            if not np.any(aligned > 0):
                raise RuntimeError("Registro RGB-depth sin pixeles validos")
            print(f"camera_{i:02d}: cobertura registrada={np.mean(aligned > 0):.1%}")
            results.append((rgb[:, :, :3].copy(), aligned))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".rgbd_", dir=destination.parent))
        for i, (rgb, depth) in enumerate(results):
            out = temporary / f"camera_{i:02d}"
            out.mkdir()
            Image.fromarray(rgb).save(out / "rgb.png")
            np.save(out / "depth.npy", depth, allow_pickle=False)
        if not completo(temporary):
            raise RuntimeError("Validacion de salida fallida")
        # Reemplazo con rollback; sin eliminar la captura anterior hasta validar.
        if destination.exists():
            old = Path(tempfile.mkdtemp(prefix=".rgbd_old_", dir=destination.parent))
            old.rmdir()
            destination.replace(old)
            try:
                temporary.replace(destination)
            except Exception:
                old.replace(destination)
                raise
            else:
                shutil.rmtree(old)
        else:
            temporary.replace(destination)
        temporary = None
        print(f"TERMINADO: {destination}; solo tres pares, depth registrado en metros")
    finally:
        timeline.stop()
        for rig in rigs:
            for sensor, names in ((rig["cs"], ["rgb"]), (rig["ds"], rig["names"])):
                try:
                    sensor.detach_annotators(names)
                    if hasattr(sensor, "destroy"):
                        sensor.destroy()
                except Exception:
                    traceback.print_exc()
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        context.new_stage()
        for _ in range(3):
            app.update()


def main():
    validar()
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": not MOSTRAR_VENTANA,
                         "renderer": "RayTracedLighting"})
    failures = []
    try:
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.sensors.experimental.rtx")
        for _ in range(5):
            app.update()
        from isaacsim.sensors.experimental.rtx import (
            RtxCamera, CameraSensor, SingleViewDepthCameraSensor)
        from src.common.paths import select_object_ids
        objects = list(select_object_ids(OBJETOS))
        if not objects:
            raise RuntimeError("No se encontraron objetos")
        for object_id in objects:
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
