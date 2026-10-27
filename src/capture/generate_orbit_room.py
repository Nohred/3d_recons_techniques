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
FRAMES_EXTRAIDOS = 300
CRF = 18
SALTAR_COMPLETOS = False
SOBRESCRIBIR_VIDEO = True
DLSS_ENABLED = False
ANTI_ALIASING = "FXAA"
RESPONSIVE_DENOISING = False
FILM_ISO = 400.0
TARGET_MEAN_MIN = 80.0
TARGET_MEAN_MAX = 150.0
PREVIEW_INDICES = (0, 120, 239)
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


def _find_setting(settings, name, candidates, *, required=False):
    observed = {}
    for path in candidates:
        try:
            observed[path] = settings.get(path)
        except Exception as error:
            observed[path] = f"{type(error).__name__}: {error}"
        if observed[path] is not None and not hasattr(observed[path], "items"):
            return path
    if required:
        raise RuntimeError(
            f"No se encontró el setting RTX para {name}. "
            f"Rutas documentadas probadas: {candidates}; valores observados: {observed}"
        )
    print(
        f"AVISO: la versión instalada no expone {name} como carb.setting; "
        f"rutas probadas={candidates}; observados={observed}"
    )
    return None


def _configure_render_settings():
    import carb.settings

    settings = carb.settings.get_settings()
    auto_exposure = _find_setting(settings, "auto exposure", (
        "/rtx/post/exposure/autoExposure",
        "/rtx/post/autoExposure/enabled",
        "/rtx/post/autoExposure",
    ))
    film_iso = _find_setting(settings, "Film ISO", (
        "/rtx/post/exposure/iso",
        "/rtx/post/filmIso",
    ))
    tone_mapping = _find_setting(settings, "tone mapping", (
        "/rtx/post/toneMapping",
        "/rtx/post/toneMapping/operator",
    ))
    anti_aliasing = _find_setting(settings, "anti-aliasing", (
        "/rtx/post/aa/algorithm",
        "/rtx/aa/algorithm",
    ))
    responsive_denoising = _find_setting(settings, "responsive denoising", (
        "/rtx/dldenoiser/responsiveDenoising",
    ))
    dlss = "/rtx/post/dlss/execMode"
    if auto_exposure:
        settings.set(auto_exposure, False)
    if film_iso:
        settings.set(film_iso, FILM_ISO)
    if tone_mapping:
        settings.set(tone_mapping, "None")
    if anti_aliasing:
        settings.set(anti_aliasing, ANTI_ALIASING)
    if responsive_denoising:
        settings.set(responsive_denoising, RESPONSIVE_DENOISING)
    if settings.get(dlss) is not None:
        settings.set(dlss, 2 if DLSS_ENABLED else 0)
    return {
        "auto_exposure": auto_exposure,
        "film_iso": film_iso,
        "tone_mapping": tone_mapping,
        "anti_aliasing": anti_aliasing,
        "responsive_denoising": responsive_denoising,
        "dlss": dlss,
        "values": {
            "auto_exposure": False,
            "film_iso": FILM_ISO,
            "tone_mapping": "None",
            "anti_aliasing": ANTI_ALIASING,
            "responsive_denoising": RESPONSIVE_DENOISING,
            "dlss_enabled": DLSS_ENABLED,
            "rt_subframes": RT_SUBFRAMES,
            "unavailable_settings": [
                name for name, path in (
                    ("auto_exposure", auto_exposure),
                    ("film_iso", film_iso),
                    ("tone_mapping", tone_mapping),
                    ("anti_aliasing", anti_aliasing),
                    ("responsive_denoising", responsive_denoising),
                ) if path is None
            ],
        },
    }


def _frame_stats(data):
    import numpy as np

    luminance = np.asarray(data[:, :, :3], dtype=np.float32).mean(axis=2)
    return {
        "mean_0_255": float(luminance.mean()),
        "saturated_percent": float(np.mean(luminance >= 255.0) * 100.0),
        "dark_percent": float(np.mean(luminance <= 16.0) * 100.0),
    }


def capture(object_id: str, app, total: int, render_settings: dict) -> None:
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import Gf, UsdGeom

    from src.capture.common.paths import orbit_output, scene_path
    from src.capture.common.scene_utils import (
        _bbox,
        camera_pose,
        framing_distance,
        is_runtime_camera_path,
        validate_scene_lighting,
    )
    from src.capture.common.room import (
        FACTOR_PARED,
        ROOM_DOME_INTENSITY,
        ROOM_LIGHT_INTENSITY,
    )

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
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Camera):
            continue
        path = str(prim.GetPath())
        if path.startswith("/OmniverseKit_"):
            # Isaac Sim crea cámaras de viewport durante la sesión.
            prim.SetActive(False)
        elif path.startswith("/Object/"):
            prim.SetActive(False)
        elif is_runtime_camera_path(path):
            prim.SetActive(False)
        else:
            raise RuntimeError(
                f"scene.usd contiene una cámara persistida fuera del objeto: {path}. "
                "Regenera la escena explícitamente."
            )
    lights = validate_scene_lighting(stage)
    print(f"{object_id}: luces activas={lights}")
    bounds = _bbox(stage.GetPrimAtPath("/Object"))
    dimensions = np.asarray(list(bounds.GetSize()), dtype=float)
    center = np.asarray(list(bounds.GetMidpoint()), dtype=float)
    radius = float(np.linalg.norm(dimensions) / 2)
    distance = framing_distance(radius, FOV_HORIZONTAL, ANCHO, ALTO, MARGEN)
    wall_distance = float(stage.GetPrimAtPath("/Room")
                          .GetAttribute("capture:wallDistance").Get())
    if wall_distance <= distance or not math.isclose(
        wall_distance / distance, FACTOR_PARED, rel_tol=1e-6
    ):
        raise RuntimeError("Las paredes no están fuera de toda la trayectoria.")
    if FRAMES_EXTRAIDOS < 2:
        raise ValueError("FRAMES_EXTRAIDOS debe ser al menos 2.")
    extracted_step = 360.0 * VUELTAS / FRAMES_EXTRAIDOS
    print(f"{object_id}: azimut entre frames extraídos={extracted_step:.3f}°")
    if extracted_step > 3.0:
        print(f"AVISO: el paso azimutal {extracted_step:.3f}° supera 3°.")

    video.parent.mkdir(parents=True, exist_ok=True)
    frames = Path(tempfile.mkdtemp(prefix=f".orbit_{object_id}_", dir=str(video.parent)))
    report_path = video.parent / "orbit_report.json"
    report = {
        "object_id": object_id,
        "configuration": {
            "resolution": [ANCHO, ALTO],
            "fps": FPS,
            "duration_seconds": DURACION_SEGUNDOS,
            "frames_rendered": total,
            "frames_extracted_by_reconstruction": FRAMES_EXTRAIDOS,
            "rt_subframes": RT_SUBFRAMES,
            "steps_per_frame": PASOS_POR_FOTOGRAMA,
            "renderer": "RayTracedLighting",
            "render_settings": render_settings,
            "room_lighting": {
                "dome_intensity": ROOM_DOME_INTENSITY,
                "rect_intensity": ROOM_LIGHT_INTENSITY,
                "rect_normalize": False,
                "rect_size_factor": 1.2,
                "rect_x_factors": [-0.55, 0.55],
            },
        },
        "preview_frames": [],
    }
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
            max(0.0001, (distance - radius) * 0.1),             max((distance + radius) * 3, wall_distance * 3)
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
        film_iso_path = render_settings["film_iso"]
        import carb.settings
        settings = carb.settings.get_settings()
        for index in range(total):
            u = index / total
            elevation = math.radians(ELEVACION_INICIAL + (ELEVACION_FINAL - ELEVACION_INICIAL) * u)
            azimuth = 2 * math.pi * VUELTAS * u
            position = center + distance * np.array([
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ])
            matrix = camera_pose(position, center)
            if not np.isclose(np.linalg.norm(matrix[:3, 3] - center), distance):
                raise RuntimeError("La pose no conserva la distancia de órbita.")
            pose.Set(Gf.Matrix4d(matrix.T.tolist()))
            for _ in range(PASOS_POR_FOTOGRAMA):
                rep.orchestrator.step(rt_subframes=RT_SUBFRAMES, delta_time=0.0,
                                      pause_timeline=True)
            data = np.asarray(rgb.get_data())
            if data.dtype != np.uint8 or data.shape[:2] != (ALTO, ANCHO):
                raise RuntimeError(f"RGB inesperado: {data.shape}, {data.dtype}")
            if index == 0:
                initial_stats = _frame_stats(data)
                if not TARGET_MEAN_MIN <= initial_stats["mean_0_255"] <= TARGET_MEAN_MAX:
                    factor = min(
                        8.0,
                        max(0.25, 120.0 / max(initial_stats["mean_0_255"], 1.0)),
                    )
                    if film_iso_path:
                        adjusted_iso = max(
                            50.0,
                            min(6400.0, FILM_ISO * factor),
                        )
                        settings.set(film_iso_path, adjusted_iso)
                        render_settings["values"]["film_iso"] = adjusted_iso
                    else:
                        for light_path in ("/Lighting/Dome",
                                           "/Lighting/RoomKey_0",
                                           "/Lighting/RoomKey_1"):
                            light = stage.GetPrimAtPath(light_path)
                            intensity = light.GetAttribute("inputs:intensity")
                            if intensity:
                                intensity.Set(float(intensity.Get()) * factor)
                        render_settings["values"]["manual_light_fallback_factor"] = factor
                        report["configuration"]["room_lighting"][
                            "manual_fallback_factor"] = factor
                    for _ in range(PASOS_POR_FOTOGRAMA):
                        rep.orchestrator.step(
                            rt_subframes=RT_SUBFRAMES, delta_time=0.0,
                            pause_timeline=True)
                    data = np.asarray(rgb.get_data())
                    if data.dtype != np.uint8 or data.shape[:2] != (ALTO, ANCHO):
                        raise RuntimeError(f"RGB inesperado: {data.shape}, {data.dtype}")
            Image.fromarray(data[:, :, :3].copy()).save(frames / f"frame_{index:06d}.png")
            preview_indices = {min(max(0, value), total - 1) for value in PREVIEW_INDICES}
            if index in preview_indices:
                stats = _frame_stats(data)
                report["preview_frames"].append({"index": index, **stats})
                print(
                    f"{object_id}: preview frame {index}: "
                    f"brillo medio={stats['mean_0_255']:.2f}, "
                    f"saturados={stats['saturated_percent']:.3f}%, "
                    f"oscuros={stats['dark_percent']:.3f}%"
                )
        means = [item["mean_0_255"] for item in report["preview_frames"]]
        report["preview_summary"] = {
            "mean_0_255": float(sum(means) / len(means)) if means else None,
            "target_range_0_255": [TARGET_MEAN_MIN, TARGET_MEAN_MAX],
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if means and not all(TARGET_MEAN_MIN <= mean <= TARGET_MEAN_MAX for mean in means):
            raise RuntimeError(
                f"El brillo medio de previews no está entre {TARGET_MEAN_MIN} y "
                f"{TARGET_MEAN_MAX}: {means}. Ajusta FILM_ISO."
            )
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
    from src.capture.common.paths import select_object_ids

    app = SimulationApp({"headless": not MOSTRAR_VENTANA, "renderer": "RayTracedLighting"})
    render_settings = _configure_render_settings()
    failures = []
    try:
        for object_id in select_object_ids(OBJETOS):
            try:
                capture(object_id, app, total, render_settings)
            except Exception:
                failures.append(object_id)
                traceback.print_exc()
        if failures:
            raise RuntimeError(f"Fallaron objetos: {failures}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
