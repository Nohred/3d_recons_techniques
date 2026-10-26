from pathlib import Path
import hashlib
import json
import math
import shutil
import subprocess
import time
import traceback

# ================= CONFIGURACION EDITABLE =================
PROYECTO = Path.home() / "Escritorio/3D_reconstruction"
CARPETA_MODELOS = PROYECTO / "3D_Models"
PROTOCOLO = "orbita_room_test_v1"
OBJETOS = None  # None procesa todos los USDZ.
MOSTRAR_VENTANA = False
ANCHO, ALTO = 1280, 720
FPS = 30
DURACION_SEGUNDOS = 8
VUELTAS = 1
ELEVACION_INICIAL = 10.0
ELEVACION_FINAL = 65.0
FOV_HORIZONTAL = 50.0
MARGEN = 1.20
INTENSIDAD_LUZ = 400.0
RT_SUBFRAMES = 8
PASOS_POR_FOTOGRAMA = 2
CRF = 18
SEMILLA = 42
CONSERVAR_PNG_RENDER = False
SALTAR_COMPLETOS = True
# Cuarto experimental, proporcional a la distancia de orbita.
FACTOR_PARED = 1.50  # Debe ser > 1: paredes fuera de la orbita.
TAMANO_TEXTURA = 2048
SEMILLA_CUARTO = 2026
INTENSIDAD_LUCES_CUARTO = 600.0
EXPORTAR_ESCENA = True
GUARDAR_PREVIAS = True
# ==========================================================


def escribir_json(ruta, datos):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(ruta):
    h = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def validar():
    if FACTOR_PARED <= 1.1 or TAMANO_TEXTURA < 256:
        raise ValueError("Usa FACTOR_PARED > 1.1 y TAMANO_TEXTURA >= 256.")
    if not CARPETA_MODELOS.is_dir():
        raise RuntimeError(f"No existe la carpeta de modelos: {CARPETA_MODELOS}")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("Instala FFmpeg: sudo apt install ffmpeg")
    if ANCHO <= 0 or ALTO <= 0 or ANCHO % 2 or ALTO % 2:
        raise ValueError("Las dimensiones deben ser positivas y pares.")
    if FPS <= 0 or DURACION_SEGUNDOS <= 0 or VUELTAS <= 0:
        raise ValueError("FPS, duracion y vueltas deben ser positivos.")
    if not 0 < FOV_HORIZONTAL < 170 or MARGEN <= 1:
        raise ValueError("Revisa el FOV y el margen de encuadre.")
    if max(abs(ELEVACION_INICIAL), abs(ELEVACION_FINAL)) >= 89:
        raise ValueError("Evita elevaciones proximas a +/-90 grados.")
    if RT_SUBFRAMES < 1 or PASOS_POR_FOTOGRAMA < 1:
        raise ValueError("Los pasos y subframes deben ser al menos 1.")
    if not 0 <= CRF <= 51:
        raise ValueError("CRF debe estar entre 0 y 51.")
    n = FPS * DURACION_SEGUNDOS
    if n < 2 or not math.isclose(n, round(n)):
        raise ValueError("FPS por duracion debe producir un numero entero de frames >= 2.")


validar()

from isaacsim import SimulationApp
app = SimulationApp({"headless": not MOSTRAR_VENTANA, "renderer": "RayTracedLighting"})

import numpy as np
from PIL import Image, ImageDraw
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
import carb.settings
import omni.usd
import omni.replicator.core as rep


def caja_mallas(prim):
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=False,
    )
    total = Gf.Range3d()
    for hijo in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if hijo.IsA(UsdGeom.Mesh):
            caja = cache.ComputeWorldBound(hijo).ComputeAlignedBox()
            if not caja.IsEmpty():
                total.UnionWith(caja)
    if total.IsEmpty():
        raise RuntimeError("No se encontraron mallas con limites validos.")
    return total


def cargar_objeto(stage, ruta):
    fuente = Usd.Stage.Open(str(ruta))
    if fuente is None:
        raise RuntimeError(f"No se pudo abrir {ruta}")
    escala = float(UsdGeom.GetStageMetersPerUnit(fuente))
    eje = str(UsdGeom.GetStageUpAxis(fuente))
    if eje not in ("Y", "Z") or escala <= 0:
        raise RuntimeError("Eje vertical o escala no soportados.")

    objeto = UsdGeom.Xform.Define(stage, "/Object")
    orientacion = UsdGeom.Xform.Define(stage, "/Object/Orientation")
    escala_prim = UsdGeom.Xform.Define(stage, "/Object/Orientation/Scale")
    escala_prim.AddScaleOp().Set(Gf.Vec3f(escala, escala, escala))
    if eje == "Y":
        orientacion.AddRotateXOp().Set(90.0)

    asset_path = "/Object/Orientation/Scale/Asset"
    default = fuente.GetDefaultPrim()
    if default:
        asset = stage.DefinePrim(asset_path)
        asset.GetReferences().AddReference(str(ruta), default.GetPath())
    else:
        UsdGeom.Xform.Define(stage, asset_path)
        for i, raiz in enumerate(fuente.GetPseudoRoot().GetChildren()):
            prim = stage.DefinePrim(f"{asset_path}/Root_{i}")
            prim.GetReferences().AddReference(str(ruta), raiz.GetPath())
    stage.Load()

    # No utilizar luces ni camaras incluidas por el autor del modelo.
    for prim in list(stage.Traverse()):
        if str(prim.GetPath()).startswith("/Object/"):
            if prim.IsA(UsdGeom.Camera) or prim.HasAPI(UsdLux.LightAPI):
                prim.SetActive(False)

    centro = caja_mallas(objeto.GetPrim()).GetMidpoint()
    objeto.AddTranslateOp().Set(Gf.Vec3d(-centro[0], -centro[1], -centro[2]))
    dimensiones = np.array(list(caja_mallas(objeto.GetPrim()).GetSize()), dtype=float)
    radio = float(np.linalg.norm(dimensiones) / 2)
    if not math.isfinite(radio) or radio <= 0:
        raise RuntimeError("Dimensiones invalidas.")
    matriz = UsdGeom.XformCache().GetLocalToWorldTransform(escala_prim.GetPrim())
    return {
        "archivo_original": str(ruta),
        "sha256_modelo": sha256(ruta),
        "meters_per_unit_original": escala,
        "unidades_explicitas": fuente.HasAuthoredMetadata("metersPerUnit"),
        "up_axis_original": eje,
        "dimensiones_xyz_m": dimensiones.tolist(),
        "radio_envolvente_m": radio,
        "asset_to_capture_matrix": np.array(matriz, dtype=float).T.tolist(),
    }


def pose_camara(posicion):
    atras = np.array(posicion, dtype=float, copy=True)
    atras /= np.linalg.norm(atras)
    derecha = np.cross([0.0, 0.0, 1.0], atras)
    derecha /= np.linalg.norm(derecha)
    arriba = np.cross(atras, derecha)
    matriz = np.eye(4)
    matriz[:3, :3] = np.column_stack([derecha, arriba, atras])
    matriz[:3, 3] = posicion
    return matriz


def codificar(carpeta, video, total):
    subprocess.run([
        shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "warning", "-n",
        "-framerate", str(FPS), "-start_number", "0",
        "-i", str(carpeta / "frame_%06d.png"),
        "-frames:v", str(total), "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", str(CRF), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(video),
    ], check=True)
    resultado = subprocess.run([
        shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries",
        "stream=nb_read_frames,width,height,avg_frame_rate", "-of", "json", str(video),
    ], check=True, capture_output=True, text=True)
    datos = json.loads(resultado.stdout)["streams"][0]
    if int(datos["nb_read_frames"]) != total:
        raise RuntimeError("El numero de frames codificados no coincide.")
    if (int(datos["width"]), int(datos["height"])) != (ANCHO, ALTO):
        raise RuntimeError("La resolucion codificada no coincide.")
    return datos



def crear_textura_local(ruta, semilla, tamano):
    """Patron irregular reproducible; no descarga assets ni texturas."""
    rng = np.random.default_rng(semilla)
    s = tamano * 2
    bases = [(178, 171, 158), (165, 181, 191), (186, 171, 185),
             (181, 189, 164), (177, 166, 151)]
    imagen = Image.new("RGB", (s, s), bases[semilla % len(bases)])
    draw = ImageDraw.Draw(imagen)
    paleta = [(48, 65, 82), (155, 77, 51), (65, 109, 94),
              (193, 155, 61), (104, 80, 130), (222, 216, 195),
              (77, 80, 88), (140, 113, 86)]
    # Detalles a varias escalas, distribuidos sin un mosaico repetido.
    for _ in range(170):
        x, y = rng.integers(0, s, size=2).tolist()
        w, h = rng.integers(max(10, s // 55), max(20, s // 7), size=2).tolist()
        color = paleta[int(rng.integers(len(paleta)))]
        tipo = int(rng.integers(3))
        if tipo == 0:
            draw.rectangle((x, y, x + w, y + h), fill=color)
        elif tipo == 1:
            draw.ellipse((x, y, x + w, y + h), fill=color)
        else:
            draw.polygon([(x, y), (x + w, y + h // 3), (x + w // 3, y + h)], fill=color)
    for _ in range(70):
        puntos = rng.integers(0, s, size=(3, 2)).tolist()
        draw.line([tuple(p) for p in puntos], fill=paleta[int(rng.integers(len(paleta)))],
                  width=int(rng.integers(2, max(4, s // 180))))
    imagen = imagen.resize((tamano, tamano), Image.Resampling.LANCZOS)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    imagen.save(ruta)


def material_texturado(stage, nombre, textura):
    material = UsdShade.Material.Define(stage, f"/Room/Materials/{nombre}")
    shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.95)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    lector = UsdShade.Shader.Define(stage, f"{material.GetPath()}/UV")
    lector.CreateIdAttr("UsdPrimvarReader_float2")
    lector.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    lector.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tex = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Texture")
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


def panel_cuarto(stage, nombre, puntos, material):
    mesh = UsdGeom.Mesh.Define(stage, f"/Room/{nombre}")
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in puntos])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    uv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    uv.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def crear_cuarto(stage, distancia, carpeta_control):
    # El objeto sigue centrado en el origen; el suelo baja hasta su base.
    caja = caja_mallas(stage.GetPrimAtPath("/Object"))
    radio = float(np.linalg.norm(np.array(list(caja.GetSize()))) / 2)
    z = float(caja.GetMin()[2]) - max(0.00001, radio * 0.002)
    limite = FACTOR_PARED * distancia
    techo_virtual = z + max(1.9 * distancia, float(caja.GetSize()[2]) * 2)
    altura_camara = distancia * math.sin(math.radians(max(ELEVACION_INICIAL, ELEVACION_FINAL)))
    techo_virtual = max(techo_virtual, altura_camara + 0.5 * distancia)
    UsdGeom.Xform.Define(stage, "/Room")
    carpeta_texturas = carpeta_control / "textures"
    paneles = {
        "Floor": [(-limite, -limite, z), (limite, -limite, z),
                  (limite, limite, z), (-limite, limite, z)],
        "North": [(-limite, limite, z), (limite, limite, z),
                  (limite, limite, techo_virtual), (-limite, limite, techo_virtual)],
        "South": [(limite, -limite, z), (-limite, -limite, z),
                  (-limite, -limite, techo_virtual), (limite, -limite, techo_virtual)],
        "East": [(limite, limite, z), (limite, -limite, z),
                 (limite, -limite, techo_virtual), (limite, limite, techo_virtual)],
        "West": [(-limite, -limite, z), (-limite, limite, z),
                 (-limite, limite, techo_virtual), (-limite, -limite, techo_virtual)],
    }
    texturas = {}
    for i, (nombre, puntos) in enumerate(paneles.items()):
        ruta = carpeta_texturas / f"{nombre.lower()}.png"
        crear_textura_local(ruta, SEMILLA_CUARTO + i * 101, TAMANO_TEXTURA)
        material = material_texturado(stage, nombre, ruta)
        panel_cuarto(stage, nombre, puntos, material)
        texturas[nombre] = {"archivo": str(ruta.resolve()), "sha256": sha256(ruta)}

    # Luces superiores fijas. Sin techo, mesa ni modificaciones al material del objeto.
    for i, x in enumerate([-0.55 * distancia, 0.55 * distancia]):
        luz = UsdLux.RectLight.Define(stage, f"/Lighting/RoomKey_{i}")
        luz.CreateIntensityAttr(INTENSIDAD_LUCES_CUARTO)
        luz.CreateNormalizeAttr(False)
        luz.CreateWidthAttr(distancia * 0.8)
        luz.CreateHeightAttr(distancia * 0.8)
        luz.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        UsdGeom.Xformable(luz.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(x, 0, techo_virtual - 0.05 * distancia))
    return {"tipo": "suelo_y_cuatro_paredes_abierto_arriba", "factor_pared": FACTOR_PARED,
            "semiancho_m": limite, "suelo_z_m": z, "altura_superior_z_m": techo_virtual,
            "semilla_texturas": SEMILLA_CUARTO, "tamano_textura": TAMANO_TEXTURA,
            "intensidad_luces_cuarto": INTENSIDAD_LUCES_CUARTO, "texturas": texturas}


def capturar(ruta):
    relativo = Path(PROTOCOLO) / ruta.stem 
    salida = PROYECTO / "data/raw" / relativo / "rgb_video"
    control = PROYECTO / "simulation_private" / relativo
    reporte_path = control / "orbit_capture.json"
    video = salida / "orbit.mp4"
    trabajo = PROYECTO / "data/_work" / relativo / "render_frames"

    if SALTAR_COMPLETOS and video.exists() and reporte_path.exists():
        anterior = json.loads(reporte_path.read_text(encoding="utf-8"))
        if anterior.get("estado") == "completo":
            print(f"OMITIDO: {ruta.name}; captura completa existente.")
            return

    
    if salida.exists() or trabajo.exists() or control.exists():
        raise RuntimeError("Ya hay una captura parcial o anterior. Cambia PROTOCOLO; no se sobrescribe.")
    salida.mkdir(parents=True)
    trabajo.mkdir(parents=True)
    reporte = {
        "estado": "iniciado", "protocolo": PROTOCOLO,
        "uso": "Control de simulacion y evaluacion. NO entrada de reconstruccion.",
        "configuracion": {
            "resolucion": [ANCHO, ALTO], "fps": FPS,
            "duracion_s": DURACION_SEGUNDOS, "vueltas": VUELTAS,
            "elevacion_inicial_deg": ELEVACION_INICIAL,
            "elevacion_final_deg": ELEVACION_FINAL,
            "fov_horizontal_deg": FOV_HORIZONTAL, "margen": MARGEN,
            "intensidad_luz": INTENSIDAD_LUZ, "rt_subframes": RT_SUBFRAMES,
            "pasos_por_fotograma": PASOS_POR_FOTOGRAMA,
            "crf": CRF, "semilla": SEMILLA,
            "renderer": "RayTracedLighting", "dlss_mode": "Quality",
            "trayectoria": "azimut=2*pi*vueltas*i/N; elevacion lineal en i/N",
        },
    }
    escribir_json(reporte_path, reporte)
    inicio = time.perf_counter()
    rgb, producto = None, None

    try:
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.SetTimeCodesPerSecond(FPS)
        stage.SetFramesPerSecond(FPS)
        rep.set_global_seed(SEMILLA)
        carb.settings.get_settings().set("/rtx/post/dlss/execMode", 2)
        rep.orchestrator.set_capture_on_play(False)

        objeto = cargar_objeto(stage, ruta)
        reporte["objeto"] = objeto
        radio = objeto["radio_envolvente_m"]
        fov_x = math.radians(FOV_HORIZONTAL)
        fov_y = 2 * math.atan(math.tan(fov_x / 2) * ALTO / ANCHO)
        distancia = MARGEN * radio / math.sin(min(fov_x, fov_y) / 2)
        reporte["distancia_camara_m"] = distancia
        escribir_json(reporte_path, reporte)
        if not objeto["unidades_explicitas"]:
            print("AVISO: el asset no declara explicitamente sus unidades.")

        reporte["cuarto"] = crear_cuarto(stage, distancia, control)
        escribir_json(reporte_path, reporte)

        luz = UsdLux.DomeLight.Define(stage, "/Lighting/Dome")
        luz.CreateIntensityAttr(INTENSIDAD_LUZ)
        luz.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        cam = UsdGeom.Camera.Define(stage, "/Camera")
        cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
        focal = 24.0
        apertura = 2 * focal * math.tan(fov_x / 2)
        cam.CreateFocalLengthAttr(focal)
        cam.CreateHorizontalApertureAttr(apertura)
        cam.CreateVerticalApertureAttr(apertura * ALTO / ANCHO)
        cam.CreateClippingRangeAttr(Gf.Vec2f(
            max(0.0001, (distancia - radio) * 0.1), max((distancia + radio) * 3, distancia * 10),
        ))
        pose_op = cam.AddTransformOp()
        producto = rep.create.render_product("/Camera", (ANCHO, ALTO))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([producto])
        for _ in range(40):
            app.update()

        total = round(FPS * DURACION_SEGUNDOS)
        print(f"{ruta.name}: {total} frames; distancia {distancia:.3f} m")
        for i in range(total):
            if not app.is_running():
                raise RuntimeError("La aplicacion se cerro antes de finalizar.")
            u = i / total
            azimut = 2 * math.pi * VUELTAS * u
            elev = math.radians(ELEVACION_INICIAL + (ELEVACION_FINAL - ELEVACION_INICIAL) * u)
            pos = distancia * np.array([
                math.cos(elev) * math.cos(azimut),
                math.cos(elev) * math.sin(azimut), math.sin(elev),
            ])
            matriz = pose_camara(pos)
            if not np.isclose(np.linalg.norm(matriz[:3, 3]), distancia):
                raise RuntimeError("La pose no conserva la distancia calculada.")
            pose_op.Set(Gf.Matrix4d(matriz.T.tolist()))
            for _ in range(PASOS_POR_FOTOGRAMA):
                rep.orchestrator.step(rt_subframes=RT_SUBFRAMES, delta_time=0.0, pause_timeline=True)
            datos = np.asarray(rgb.get_data())
            if datos.ndim != 3 or datos.shape[:2] != (ALTO, ANCHO) or datos.shape[2] < 3 or datos.dtype != np.uint8:
                raise RuntimeError(f"RGB inesperado: {datos.shape}, {datos.dtype}")
            Image.fromarray(datos[:, :, :3].copy()).save(trabajo / f"frame_{i:06d}.png")
            if GUARDAR_PREVIAS and i in {0, total // 4, total // 2, 3 * total // 4}:
                previa = control / "previews" / f"view_{i:06d}.png"
                previa.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(datos[:, :, :3].copy()).save(previa)
            if i % FPS == 0 or i == total - 1:
                print(f"  {i + 1}/{total}")

        if EXPORTAR_ESCENA:
            escena = control / "scene.usda"
            stage.GetRootLayer().Export(str(escena.resolve()))
            reporte["escena_exportada"] = str(escena.resolve())
        reporte["video_verificacion"] = codificar(trabajo, video, total)
        reporte["sha256_video"] = sha256(video)
        reporte["tiempo_generacion_s"] = time.perf_counter() - inicio
        if not CONSERVAR_PNG_RENDER:
            shutil.rmtree(trabajo)
        reporte["estado"] = "completo"
        escribir_json(reporte_path, reporte)
        print(f"TERMINADO: {video}")
    except Exception:
        reporte["estado"] = "error"
        reporte["error"] = traceback.format_exc()
        escribir_json(reporte_path, reporte)
        raise
    finally:
        if rgb is not None:
            rgb.detach()
        if producto is not None:
            producto.destroy()
        for _ in range(3):
            app.update()


try:
    archivos = (
        sorted(p for p in CARPETA_MODELOS.iterdir() if p.suffix.lower() == ".usdz")
        if OBJETOS is None else [CARPETA_MODELOS / nombre for nombre in OBJETOS]
    )
    if not archivos:
        raise RuntimeError("No se encontraron objetos.")
    fallos = []
    for archivo in archivos:
        try:
            capturar(archivo.resolve())
        except Exception as error:
            print(f"ERROR {archivo.name}: {error}")
            traceback.print_exc()
            fallos.append(archivo.name)
    print(f"Fin. Objetos intentados: {len(archivos)}. Fallos: {fallos}")
    if fallos:
        raise RuntimeError("Algunas capturas fallaron; revisa los reportes privados.")
finally:
    app.close()
