"""Captura de calibracion por objeto: tablero ChArUco en el piso + las 3 camaras RGB-D del objeto.

Para cada objeto abre su data/<obj>/scene/scene.usd (mismo cuarto y luces), DESACTIVA el objeto en la
capa de sesion (no se modifica scene.usd), coloca el tablero en el piso bajo el centro del objeto y
renderiza las MISMAS tres camaras que usa generate_rgbd_three_cameras.py.

Las camaras dependen de cada objeto (distancia segun su radio), por eso la calibracion se genera por objeto.
Ejecutar con python.sh de Isaac Sim. Antes (una vez):  python src/reconstruct/cameras/charuco_board.py
"""
from pathlib import Path
import json
import math
import shutil
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[3]
for ruta in (ROOT, Path(__file__).resolve().parent):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

# ================= CONFIGURACION EDITABLE =================
OBJETOS = ["sneaker"]          # None: todos los USDZ (cada uno necesita data/<obj>/scene/scene.usd)
MOSTRAR_VENTANA = False

# --- DEBEN COINCIDIR con generate_rgbd_three_cameras.py (camaras identicas) ---
ANCHO, ALTO = 1280, 720
AZIMUTS = (0.0, 120.0, 240.0)
ELEVACION = 25.0
FOV_HORIZONTAL = 50.0
MARGEN = 1.20
DISTANCIA_MIN_OPERATIVA_M = None   # None = camaras IDEAL. Con D455 usa el mismo valor (p. ej. 0.60) y
                                   # copia el FOV horizontal que imprime esa captura.
# -----------------------------------------------------------------------------

ESCALA_CALIBRACION = 1         # 2 = renderiza el tablero al doble de resolucion (mismas poses; K se escala)
GUARDAR_PROFUNDIDAD = False    # guarda tambien distance_to_image_plane de la calibracion

# Tablero
ANCHO_PLANO_MAX_FRACCION_D = 0.55   # ancho maximo del plano (con margen) como fraccion de la distancia de camara
HOLGURA_MIN_PX = 30                 # el plano completo debe quedar a esta distancia del borde de la imagen
ELEVAR_TABLERO_M = 0.001            # separacion minima sobre el piso (evita z-fighting)
VOLTEAR_V_TEXTURA = False           # ponlo en True si el tablero sale en espejo (la verificacion lo avisara)
MIN_PX_CASILLA = 30                 # aviso si una casilla mide menos (pixeles de la imagen de calibracion)
EXIGIR_MIN_PX = False

# Verificacion con OpenCV (si cv2.aruco esta disponible en el Python de Isaac Sim)
VERIFICAR_DETECCION = True
MIN_ESQUINAS_DETECTADAS = 12
EXIGIR_DETECCION = True

# Render
RT_SUBFRAMES = 8
PASOS_CALENTAMIENTO = 40
PASOS_CAPTURA = 8
ANTI_ALIASING_OP = 2               # /rtx/post/aa/op: 0 Off, 2 FXAA, 3 DLSS, 4 DLAA; None = no tocar
APAGAR_AUTOEXPOSICION = True       # /rtx/post/histogram/enabled = False
SOBRESCRIBIR = True
# ==========================================================


def _validar_config() -> None:
    if len(AZIMUTS) != 3 or not all(math.isfinite(a) for a in AZIMUTS):
        raise ValueError("Se requieren exactamente tres azimuts finitos.")
    if not 0 < ELEVACION < 89 or not 0 < FOV_HORIZONTAL < 170 or MARGEN <= 1:
        raise ValueError("Elevacion, FOV o margen invalidos (la elevacion debe ser > 0 para ver el piso).")
    if ANCHO % 2 or ALTO % 2 or ESCALA_CALIBRACION < 1 or int(ESCALA_CALIBRACION) != ESCALA_CALIBRACION:
        raise ValueError("ANCHO/ALTO pares y ESCALA_CALIBRACION entera >= 1.")
    if not 0 < ANCHO_PLANO_MAX_FRACCION_D < 2 or min(RT_SUBFRAMES, PASOS_CALENTAMIENTO, PASOS_CAPTURA) < 1:
        raise ValueError("Revisa ANCHO_PLANO_MAX_FRACCION_D y los pasos de render.")


def _intrinsecos(ancho: int, alto: int) -> dict:
    fx = ancho / (2.0 * math.tan(math.radians(FOV_HORIZONTAL) / 2.0))
    return {"fx": fx, "fy": fx, "cx": ancho / 2.0, "cy": alto / 2.0, "ancho": ancho, "alto": alto}


def _matriz_K(k: dict):
    import numpy as np
    return np.array([[k["fx"], 0.0, k["cx"]], [0.0, k["fy"], k["cy"]], [0.0, 0.0, 1.0]])


def _configurar_camara(camera, matrix, distance: float, radius: float) -> None:
    """Identica a _configure_camera de generate_rgbd_three_cameras.py (modo IDEAL)."""
    from pxr import Gf, UsdGeom
    focal = 24.0
    aperture = 2 * focal * math.tan(math.radians(FOV_HORIZONTAL) / 2)
    camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    camera.CreateFocalLengthAttr(focal)
    camera.CreateHorizontalApertureAttr(aperture)
    camera.CreateVerticalApertureAttr(aperture * ALTO / ANCHO)
    camera.CreateClippingRangeAttr(Gf.Vec2f(max(0.0001, (distance - radius) * 0.1),
                                            max((distance + radius) * 3, distance * 10)))
    transform = UsdGeom.Xformable(camera.GetPrim())
    transform.ClearXformOpOrder()
    transform.SetResetXformStack(True)
    transform.AddTransformOp().Set(Gf.Matrix4d(matrix.T.tolist()))


def _altura_suelo(stage, z_respaldo: float):
    """Cara superior del suelo: prims bajo /Room cuyo nombre contiene 'floor'. Respaldo: base del objeto."""
    from pxr import Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                              [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False)
    zs = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/Room")):
        if "floor" in prim.GetName().lower() and prim.IsA(UsdGeom.Gprim):
            caja = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if not caja.IsEmpty():
                zs.append(float(caja.GetMax()[2]))
    return (max(zs), True) if zs else (z_respaldo, False)


def _material_tablero(stage, textura: Path):
    from pxr import Sdf, UsdShade
    base = "/CalibrationBoard/Material"
    material = UsdShade.Material.Define(stage, base)
    shader = UsdShade.Shader.Define(stage, f"{base}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    lector = UsdShade.Shader.Define(stage, f"{base}/UV")
    lector.CreateIdAttr("UsdPrimvarReader_float2")
    lector.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    lector.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tex = UsdShade.Shader.Define(stage, f"{base}/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(textura.resolve())))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(lector.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _crear_tablero(stage, esquinas, textura: Path) -> None:
    """Plano con el tablero. Vertices: abajo-izq, abajo-der, arriba-der, arriba-izq (visto desde arriba)."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    UsdGeom.Xform.Define(stage, "/CalibrationBoard")
    malla = UsdGeom.Mesh.Define(stage, "/CalibrationBoard/Plane")
    malla.CreatePointsAttr([Gf.Vec3f(*[float(v) for v in p]) for p in esquinas])
    malla.CreateFaceVertexCountsAttr([4])
    malla.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    malla.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    malla.CreateDoubleSidedAttr(True)
    malla.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    malla.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
    if VOLTEAR_V_TEXTURA:
        uv = [(u, 1 - v) for u, v in uv]
    primvar = UsdGeom.PrimvarsAPI(malla.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    primvar.Set([Gf.Vec2f(float(u), float(v)) for u, v in uv])
    UsdShade.MaterialBindingAPI.Apply(malla.GetPrim()).Bind(_material_tablero(stage, textura))


def _leer_rgb(anotador, ancho: int, alto: int):
    import numpy as np
    color = np.asarray(anotador.get_data())
    if color.dtype != np.uint8 or color.ndim != 3 or color.shape[:2] != (alto, ancho) or color.shape[2] < 3:
        raise RuntimeError(f"RGB inesperado: shape={color.shape}, dtype={color.dtype}")
    return color[:, :, :3].copy()


def _leer_profundidad(anotador, ancho: int, alto: int):
    import numpy as np
    d = np.asarray(anotador.get_data())
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[:, :, 0]
    if d.shape != (alto, ancho):
        raise RuntimeError(f"Profundidad inesperada: shape={d.shape}")
    d = d.astype(np.float32, copy=True)
    d[~np.isfinite(d) | (d <= 0)] = 0.0
    return d


def _reemplazar(temporal: Path, destino: Path) -> None:
    """Reemplazo atomico con rollback (mismo esquema que generate_rgbd_three_cameras.py)."""
    if destino.exists():
        respaldo = Path(tempfile.mkdtemp(prefix=".cal_old_", dir=str(destino.parent)))
        respaldo.rmdir()
        destino.replace(respaldo)
        try:
            temporal.replace(destino)
        except Exception:
            respaldo.replace(destino)
            raise
        shutil.rmtree(respaldo)
    else:
        temporal.replace(destino)


def _ajustes_render() -> dict:
    """Aplica y reporta ajustes RTX. Debe llamarse ANTES de crear los render products."""
    import carb.settings
    s = carb.settings.get_settings()
    if ANTI_ALIASING_OP is not None:
        s.set("/rtx/post/aa/op", int(ANTI_ALIASING_OP))
    if APAGAR_AUTOEXPOSICION:
        s.set("/rtx/post/histogram/enabled", False)
    leidos = {"/rtx/post/aa/op": s.get("/rtx/post/aa/op"),
              "/rtx/post/histogram/enabled": s.get("/rtx/post/histogram/enabled"),
              "/rtx/post/tonemap/filmIso": s.get("/rtx/post/tonemap/filmIso")}
    print(f"[render] {leidos}")
    return {k: (None if v is None else (float(v) if isinstance(v, float) else v)) for k, v in leidos.items()}


def capturar(object_id: str, app) -> None:
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import UsdGeom

    import charuco_board as cb
    from src.capture.common.paths import rgbd_output, scene_path
    from src.capture.common.scene_utils import _bbox, camera_pose, framing_distance, validate_scene_lighting

    rgbd_dir = rgbd_output(object_id, 0).parent
    raw_dir = rgbd_dir.parent
    if raw_dir.name != "raw":
        raise RuntimeError(f"Estructura inesperada de rutas: {rgbd_dir}")
    obj_dir = raw_dir.parent
    destino = raw_dir / "calibration"
    privado = obj_dir / "simulation_private"
    if destino.exists() and not SOBRESCRIBIR:
        raise RuntimeError(f"Ya existe {destino}; activa SOBRESCRIBIR.")
    escena = scene_path(object_id)
    if not escena.is_file():
        raise RuntimeError(f"Falta {escena}. Ejecuta prepare_scenes.py.")
    textura = cb.asegurar_imagen()

    contexto = omni.usd.get_context()
    if not contexto.open_stage(str(escena.resolve())):
        raise RuntimeError(f"No se pudo abrir {escena}")
    stage = contexto.get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())  # nada de lo siguiente se escribe en scene.usd

    if str(UsdGeom.GetStageUpAxis(stage)) != "Z" or not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        raise RuntimeError("scene.usd debe estar en metros con eje Z.")
    sala, objeto = stage.GetPrimAtPath("/Room"), stage.GetPrimAtPath("/Object")
    if not sala or not objeto:
        raise RuntimeError("scene.usd no contiene /Room y /Object.")
    pared_attr = sala.GetAttribute("capture:wallDistance")
    if not pared_attr or not pared_attr.Get():
        raise RuntimeError("scene.usd no contiene un cuarto dinamico valido (capture:wallDistance).")
    pared = float(pared_attr.Get())
    luces = validate_scene_lighting(stage)
    print(f"{object_id}: luces activas={luces}")

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            ruta = str(prim.GetPath())
            if ruta.startswith(("/OmniverseKit_", "/Object/")):
                prim.SetActive(False)
            else:
                raise RuntimeError(f"Camara persistida inesperada: {ruta}. Regenera la escena.")

    # --- Geometria identica a la captura RGB-D (se mide ANTES de ocultar el objeto) ---
    caja = _bbox(objeto)
    dimensiones = np.asarray(list(caja.GetSize()), dtype=float)
    centro = np.asarray(list(caja.GetMidpoint()), dtype=float)
    radio = float(np.linalg.norm(dimensiones) / 2)
    if not np.isfinite(radio) or radio <= 0:
        raise RuntimeError("Bounding box invalido.")
    distancia = framing_distance(radio, FOV_HORIZONTAL, ANCHO, ALTO, MARGEN)
    if DISTANCIA_MIN_OPERATIVA_M is not None:
        distancia = max(distancia, float(DISTANCIA_MIN_OPERATIVA_M) + radio)
    if distancia >= pared:
        raise RuntimeError(f"La captura queda fuera del cuarto (distancia={distancia:.3f} m, paredes={pared:.3f} m).")
    objeto.SetActive(False)  # solo en la capa de sesion

    z_suelo, suelo_encontrado = _altura_suelo(stage, float(caja.GetMin()[2]))
    if not suelo_encontrado:
        print(f"AVISO {object_id}: no se encontro un prim 'floor' bajo /Room; se usa la base del objeto.")
    z_tablero = z_suelo + max(ELEVAR_TABLERO_M, 0.0)
    centro_xy = (float(centro[0]), float(centro[1]))

    poses = []
    for azimut in AZIMUTS:
        el, az = math.radians(ELEVACION), math.radians(azimut)
        posicion = centro + distancia * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        poses.append(np.asarray(camera_pose(posicion, centro), dtype=float))

    esc = int(ESCALA_CALIBRACION)
    ancho_c, alto_c = ANCHO * esc, ALTO * esc
    K_cal = _intrinsecos(ancho_c, alto_c)
    K_rgbd = _intrinsecos(ANCHO, ALTO)
    long_casilla, metricas = cb.elegir_longitud_casilla(
        centro_xy, z_tablero, poses, _matriz_K(K_cal), ancho_c, alto_c,
        ancho_plano_max_m=ANCHO_PLANO_MAX_FRACCION_D * distancia, holgura_min_px=HOLGURA_MIN_PX * esc)
    for i, m in enumerate(metricas):
        print(f"{object_id}: camara {i}: casilla={m['px_casilla_min']:.1f}px "
              f"(x={m['px_casilla_x']:.1f}, y={m['px_casilla_y']:.1f}), holgura={m['holgura_px']:.0f}px, "
              f"angulo con la normal={m['angulo_con_normal_deg']:.1f} deg")
    pequenas = [i for i, m in enumerate(metricas) if m["px_casilla_min"] < MIN_PX_CASILLA]
    if pequenas:
        msg = (f"{object_id}: casillas < {MIN_PX_CASILLA}px en camaras {pequenas}; la deteccion puede fallar. "
               "Prueba ESCALA_CALIBRACION = 2.")
        if EXIGIR_MIN_PX:
            raise RuntimeError(msg)
        print("AVISO " + msg)
    print(f"{object_id}: distancia={distancia:.4f} m, paredes={pared:.4f} m, "
          f"casilla={long_casilla * 1000:.1f} mm, tablero {cb.CASILLAS_X}x{cb.CASILLAS_Y}")

    ajustes = _ajustes_render()
    _crear_tablero(stage, cb.esquinas_plano_mundo(centro_xy, z_tablero, long_casilla), textura)

    recursos = {"productos": [], "anotadores": []}
    parejas, temporal = [], None
    estado = "completo"
    try:
        for i, c2w in enumerate(poses):
            camara = UsdGeom.Camera.Define(stage, f"/CaptureCalibration/Camera_{i:02d}")
            _configurar_camara(camara, c2w, distancia, radio)
            producto = rep.create.render_product(str(camara.GetPath()), (ancho_c, alto_c))
            rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb.attach([producto])
            recursos["productos"].append(producto)
            recursos["anotadores"].append(rgb)
            prof = None
            if GUARDAR_PROFUNDIDAD:
                prof = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
                prof.attach([producto])
                recursos["anotadores"].append(prof)
            parejas.append((rgb, prof))

        rep.orchestrator.set_capture_on_play(False)
        for _ in range(PASOS_CALENTAMIENTO):
            app.update()
        for _ in range(PASOS_CAPTURA):
            if not app.is_running():
                raise RuntimeError("Isaac Sim se cerro durante la captura.")
            rep.orchestrator.step(rt_subframes=RT_SUBFRAMES, delta_time=0.0, pause_timeline=True)

        datos = [(_leer_rgb(r, ancho_c, alto_c), None if p is None else _leer_profundidad(p, ancho_c, alto_c))
                 for r, p in parejas]

        # --- Verificacion de deteccion (opcional) ---
        deteccion = []
        if VERIFICAR_DETECCION:
            try:
                for color, _ in datos:
                    deteccion.append(cb.detectar(color))
            except Exception as error:
                deteccion = []
                print(f"AVISO: no se pudo verificar la deteccion ({error}).")
        for i, d in enumerate(deteccion):
            print(f"{object_id}: camara {i}: marcadores={d['marcadores']}, esquinas={d['esquinas']}/{d['esquinas_totales']}")
        if deteccion and any(d["esquinas"] < MIN_ESQUINAS_DETECTADAS for d in deteccion):
            estado = "verificacion_fallida"

        # --- Escritura ---
        destino.parent.mkdir(parents=True, exist_ok=True)
        temporal = Path(tempfile.mkdtemp(prefix=".cal_", dir=str(destino.parent)))
        for i, (color, prof) in enumerate(datos):
            carpeta = temporal / f"camera_{i:02d}"
            carpeta.mkdir()
            Image.fromarray(color).save(carpeta / "rgb.png")
            if prof is not None:
                np.save(carpeta / "depth.npy", prof, allow_pickle=False)
        entradas = {
            "descripcion": "Entradas publicas de la calibracion (equivalen a lo que conoce un usuario real).",
            "objeto": object_id,
            "resolucion_calibracion": [ancho_c, alto_c],
            "escala_respecto_a_rgbd": esc,
            "K_calibracion": K_cal,
            "K_rgbd": K_rgbd,
            "distorsion": [0.0, 0.0, 0.0, 0.0, 0.0],
            "camaras": [f"camera_{i:02d}" for i in range(3)],
            "tablero": cb.especificacion(long_casilla),
            "tablero_png_sha256": cb.sha256_archivo(textura),
        }
        (temporal / "calibration_inputs.json").write_text(json.dumps(entradas, indent=2, ensure_ascii=False), encoding="utf-8")
        _reemplazar(temporal, destino)
        temporal = None

        F = np.diag([1.0, -1.0, -1.0, 1.0])  # camara USD (mira a -Z, Y arriba) -> camara OpenCV
        verdad = {
            "descripcion": "Verdad del simulador. NO es entrada de la reconstruccion; solo evaluacion.",
            "objeto": object_id, "estado": estado,
            "T_mundo_tablero": cb.marco_tablero_en_mundo(centro_xy, z_tablero, long_casilla).tolist(),
            "T_mundo_camara_usd": [p.tolist() for p in poses],
            "T_mundo_camara_opencv": [(p @ F).tolist() for p in poses],
            "centro_objeto": centro.tolist(), "radio_envolvente_m": radio, "distancia_camara_m": distancia,
            "pared_m": pared, "z_suelo": z_suelo, "suelo_encontrado": suelo_encontrado, "z_tablero": z_tablero,
            "metricas_vista": metricas, "deteccion": deteccion, "luces": list(luces),
            "ajustes_render": ajustes,
            "config": {"AZIMUTS": list(AZIMUTS), "ELEVACION": ELEVACION, "FOV_HORIZONTAL": FOV_HORIZONTAL,
                       "MARGEN": MARGEN, "ANCHO": ANCHO, "ALTO": ALTO,
                       "DISTANCIA_MIN_OPERATIVA_M": DISTANCIA_MIN_OPERATIVA_M,
                       "ESCALA_CALIBRACION": esc, "VOLTEAR_V_TEXTURA": VOLTEAR_V_TEXTURA},
        }
        privado.mkdir(parents=True, exist_ok=True)
        (privado / "calibration_ground_truth.json").write_text(json.dumps(verdad, indent=2), encoding="utf-8")
        print(f"TERMINADO: {destino} (estado={estado})")
        if estado != "completo" and EXIGIR_DETECCION:
            raise RuntimeError(
                f"{object_id}: el tablero no se detecto bien en alguna camara (< {MIN_ESQUINAS_DETECTADAS} esquinas). "
                "Revisa las imagenes; si el tablero sale en espejo, VOLTEAR_V_TEXTURA = True; "
                "si se ve muy pequeno, ESCALA_CALIBRACION = 2.")
    finally:
        if temporal is not None:
            shutil.rmtree(temporal, ignore_errors=True)
        for anotador in recursos["anotadores"]:
            try:
                anotador.detach()
            except Exception:
                traceback.print_exc()
        for producto in recursos["productos"]:
            try:
                producto.destroy()
            except Exception:
                traceback.print_exc()
        contexto.new_stage()
        for _ in range(3):
            app.update()


def main() -> None:
    _validar_config()
    from isaacsim import SimulationApp
    from src.capture.common.paths import select_object_ids

    app = SimulationApp({"headless": not MOSTRAR_VENTANA, "renderer": "RayTracedLighting"})
    fallos = []
    try:
        for object_id in select_object_ids(OBJETOS):
            try:
                capturar(object_id, app)
            except Exception:
                fallos.append(object_id)
                traceback.print_exc()
        if fallos:
            raise RuntimeError(f"Fallaron objetos: {fallos}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
