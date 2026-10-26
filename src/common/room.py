from __future__ import annotations

import hashlib
import math
from pathlib import Path


FACTOR_PARED = 1.25
TEXTURE_SIZE = 2048
TEXTURE_SEED = 2026
ROOM_DOME_INTENSITY = 800.0
ROOM_LIGHT_INTENSITY = 900.0
ROOM_LIGHT_NORMALIZE = False
ROOM_TEXTURE_NAME = "generated_room.png"


def texture_path() -> Path:
    from src.common.paths import SHARED_ASSETS_ROOT

    return SHARED_ASSETS_ROOT / "environment" / "textures" / ROOM_TEXTURE_NAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_texture(path: Path, size: int = TEXTURE_SIZE) -> str:
    from PIL import Image, ImageDraw
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(TEXTURE_SEED)
    image = Image.new("RGB", (size, size), (112, 118, 125))
    draw = ImageDraw.Draw(image)
    palette = [(28, 43, 58), (183, 62, 42), (37, 112, 93),
               (218, 164, 47), (116, 65, 139), (226, 218, 190),
               (64, 70, 79), (154, 104, 61)]
    tile = size // 5
    for region in range(5):
        left = region * tile
        right = size if region == 4 else (region + 1) * tile
        base = tuple(int(value) for value in rng.integers(85, 155, size=3))
        draw.rectangle((left, 0, right, size), fill=base)
        for _ in range(42):
            x = int(rng.integers(left, max(left + 1, right)))
            y = int(rng.integers(0, size))
            width = int(rng.integers(max(24, tile // 10), max(32, tile // 2)))
            height = int(rng.integers(max(24, size // 14), max(40, size // 4)))
            points = [
                (x, y),
                (x + width, y + int(height * rng.uniform(0.2, 0.8))),
                (x + int(width * rng.uniform(0.3, 0.9)), y + height),
                (x - int(width * rng.uniform(0.0, 0.4)),
                 y + int(height * rng.uniform(0.4, 1.0))),
            ]
            draw.polygon(points, fill=palette[int(rng.integers(len(palette)))])
        for _ in range(3):
            x = int(rng.integers(left, max(left + 1, right)))
            y = int(rng.integers(0, size))
            radius = int(rng.integers(max(18, tile // 12), max(30, tile // 5)))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         fill=palette[int(rng.integers(len(palette)))])
        noise = rng.normal(0, 12, size=(size, max(1, right - left), 1))
        crop = np.asarray(image.crop((left, 0, right, size)), dtype=np.float32)
        crop = np.clip(crop + noise, 0, 255).astype(np.uint8)
        image.paste(Image.fromarray(crop), (left, 0))
    image.save(path)
    return sha256(path)


def _material(stage, name: str, texture: Path, uv_scale: tuple[float, float],
              rotation: float, offset: tuple[float, float]):
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, f"/Room/Materials/{name}")
    shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.95)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.0)
    reader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    transform = UsdShade.Shader.Define(stage, f"{material.GetPath()}/UVTransform")
    transform.CreateIdAttr("UsdTransform2d")
    transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(uv_scale)
    transform.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(rotation)
    transform.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set(offset)
    transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result")
    transform.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture_shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Texture")
    texture_shader.CreateIdAttr("UsdUVTexture")
    texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(texture.resolve())))
    texture_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    texture_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        transform.ConnectableAPI(), "result")
    texture_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture_shader.ConnectableAPI(), "rgb")
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _panel(stage, name, points, material):
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    mesh = UsdGeom.Mesh.Define(stage, f"/Room/{name}")
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    primvars.CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    ).Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def create_room(stage, distance: float, dimensions, floor_z: float) -> dict:
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("La distancia de órbita debe ser positiva.")
    wall = FACTOR_PARED * distance
    height = floor_z + max(1.9 * distance, float(dimensions[2]) * 2)
    texture = texture_path()
    texture_hash = ensure_texture(texture)
    UsdGeom.Xform.Define(stage, "/Room")
    panels = {
        "Floor": [(-wall, -wall, floor_z), (wall, -wall, floor_z),
                  (wall, wall, floor_z), (-wall, wall, floor_z)],
        "North": [(-wall, wall, floor_z), (wall, wall, floor_z),
                  (wall, wall, height), (-wall, wall, height)],
        "South": [(wall, -wall, floor_z), (-wall, -wall, floor_z),
                  (-wall, -wall, height), (wall, -wall, height)],
        "East": [(wall, wall, floor_z), (wall, -wall, floor_z),
                 (wall, -wall, height), (wall, wall, height)],
        "West": [(-wall, -wall, floor_z), (-wall, wall, floor_z),
                 (-wall, wall, height), (-wall, -wall, height)],
    }
    uv_transforms = [
        ((0.2, 1.0), 0.0, (0.0, 0.0)),
        ((0.2, 1.0), 90.0, (0.2, 0.07)),
        ((0.2, 1.0), 180.0, (0.4, 0.19)),
        ((0.2, 1.0), 270.0, (0.6, 0.31)),
        ((0.2, 1.0), 35.0, (0.8, 0.43)),
    ]
    for (name, points), transform in zip(panels.items(), uv_transforms):
        _panel(stage, name, points, _material(stage, name, texture, *transform))

    lighting = UsdGeom.Xform.Define(stage, "/Lighting")
    dome = UsdLux.DomeLight.Define(stage, "/Lighting/Dome")
    dome.CreateIntensityAttr(ROOM_DOME_INTENSITY)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    for index, x in enumerate((-0.55 * distance, 0.55 * distance)):
        light = UsdLux.RectLight.Define(stage, f"/Lighting/RoomKey_{index}")
        light.CreateIntensityAttr(ROOM_LIGHT_INTENSITY)
        light.CreateNormalizeAttr(ROOM_LIGHT_NORMALIZE)
        light.CreateWidthAttr(1.2 * distance)
        light.CreateHeightAttr(1.2 * distance)
        light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(x, 0.0, height - 0.05 * distance))
    room = stage.GetPrimAtPath("/Room")
    room.CreateAttribute("capture:wallDistance", Sdf.ValueTypeNames.Float).Set(wall)
    room.CreateAttribute("capture:floorZ", Sdf.ValueTypeNames.Float).Set(floor_z)
    room.CreateAttribute("capture:ceilingZ", Sdf.ValueTypeNames.Float).Set(height)
    room.CreateAttribute("capture:texturePath", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(texture.resolve())))
    room.CreateAttribute("capture:textureSha256", Sdf.ValueTypeNames.String).Set(
        texture_hash)
    return {"factor_pared": FACTOR_PARED, "wall_distance_m": wall,
            "floor_z_m": floor_z, "height_m": height,
            "texture": str(texture.resolve()), "texture_sha256": texture_hash}
