from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np


def _bbox(prim):
    from pxr import Gf, Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=False,
    )
    result = Gf.Range3d()
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if child.IsA(UsdGeom.Mesh):
            bound = cache.ComputeWorldBound(child).ComputeAlignedBox()
            if not bound.IsEmpty():
                result.UnionWith(bound)
    if result.IsEmpty():
        raise RuntimeError("No se encontraron mallas con límites válidos.")
    return result


def add_object(stage, model_path: Path):
    from pxr import Gf, Usd, UsdGeom, UsdLux

    source = Usd.Stage.Open(str(model_path))
    if source is None:
        raise RuntimeError(f"No se pudo abrir {model_path}")
    meters = float(UsdGeom.GetStageMetersPerUnit(source))
    up_axis = str(UsdGeom.GetStageUpAxis(source))
    if meters <= 0 or up_axis not in {"Y", "Z"}:
        raise RuntimeError(f"Unidades/eje no soportados en {model_path}")
    obj = UsdGeom.Xform.Define(stage, "/Object")
    orientation = UsdGeom.Xform.Define(stage, "/Object/Orientation")
    scale = UsdGeom.Xform.Define(stage, "/Object/Orientation/Scale")
    scale.AddScaleOp().Set(Gf.Vec3f(meters, meters, meters))
    if up_axis == "Y":
        orientation.AddRotateXOp().Set(90.0)
    asset_path = "/Object/Orientation/Scale/Asset"
    layer_dir = Path(stage.GetRootLayer().realPath).parent
    reference_path = os.path.relpath(model_path.resolve(), layer_dir)
    default = source.GetDefaultPrim()
    if default:
        stage.DefinePrim(asset_path).GetReferences().AddReference(
            reference_path, default.GetPath()
        )
    else:
        for index, root in enumerate(source.GetPseudoRoot().GetChildren()):
            stage.DefinePrim(f"{asset_path}/Root_{index}").GetReferences().AddReference(
                reference_path, root.GetPath()
            )
    stage.Load()
    for prim in list(stage.TraverseAll()):
        if str(prim.GetPath()).startswith("/Object/") and (
            prim.IsA(UsdGeom.Camera) or prim.HasAPI(UsdLux.LightAPI)
        ):
            prim.SetActive(False)
    bounds = _bbox(obj.GetPrim())
    min_point = bounds.GetMin()
    center = bounds.GetMidpoint()
    obj.AddTranslateOp().Set(Gf.Vec3d(-center[0], -center[1], -min_point[2]))
    placed = _bbox(obj.GetPrim())
    dimensions = np.asarray(list(placed.GetSize()), dtype=float)
    radius = float(np.linalg.norm(dimensions) / 2)
    if not math.isfinite(radius) or radius <= 0:
        raise RuntimeError("Dimensiones inválidas del objeto.")
    return {"center": np.asarray(list(placed.GetMidpoint()), dtype=float),
            "dimensions": dimensions, "radius": radius,
            "floor_z": float(placed.GetMin()[2])}


def create_scene(stage, model_path: Path, room_path: Path | None = None,
                 *, distance: float | None = None):
    from pxr import UsdGeom

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    object_info = add_object(stage, model_path)
    if distance is None:
        distance = framing_distance(object_info["radius"], 50.0, 1280, 720, 1.2)
    from src.capture.common.room import create_room
    object_info["room"] = create_room(
        stage, distance, object_info["dimensions"], object_info["floor_z"])
    return object_info


def is_runtime_camera_path(path: str) -> bool:
    """Cámaras que Isaac Sim crea para el viewport, no para la escena."""
    return path == "/OmniverseKit_Persp" or path.startswith("/Render/")


def validate_scene_lighting(stage) -> list[str]:
    from pxr import UsdLux

    lights = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdLux.LightAPI) and prim.IsActive()
    ]
    if not lights:
        raise RuntimeError(
            "La escena no contiene luces activas. Regenera las escenas con "
            "REBUILD_SCENES=True."
        )
    return lights


def framing_distance(radius: float, horizontal_fov_deg: float,
                     width: int, height: int, margin: float) -> float:
    fx = math.radians(horizontal_fov_deg)
    fy = 2 * math.atan(math.tan(fx / 2) * height / width)
    return margin * radius / math.sin(min(fx, fy) / 2)


def camera_pose(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    from numpy.linalg import norm

    position_copy = np.array(position, dtype=float, copy=True)
    backward = position_copy - np.asarray(target, dtype=float)
    backward /= norm(backward)
    right = np.cross([0.0, 0.0, 1.0], backward)
    right /= norm(right)
    up = np.cross(backward, right)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([right, up, backward])
    matrix[:3, 3] = position_copy
    return matrix


def validate_room_fit(dimensions: np.ndarray, center: np.ndarray,
                      distance: float, room_width: float, room_depth: float,
                      room_height: float) -> None:
    if dimensions[0] > room_width or dimensions[1] > room_depth:
        raise RuntimeError("El objeto no cabe en el cuarto fijo.")
    if center[2] + dimensions[2] / 2 > room_height:
        raise RuntimeError("El objeto supera la altura del cuarto fijo.")
    if distance >= min(room_width, room_depth) / 2:
        raise RuntimeError("La trayectoria de cámaras excede el cuarto fijo.")
