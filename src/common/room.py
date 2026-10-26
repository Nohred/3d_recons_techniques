from __future__ import annotations

from pathlib import Path


# Configuración única y fija del entorno, en metros. El objeto nunca modifica esto.
ROOM_WIDTH_M = 6.0
ROOM_DEPTH_M = 6.0
ROOM_HEIGHT_M = 4.0
FLOOR_Z_M = 0.0
TEXTURE_SIZE = 1024
TEXTURE_SEED = 2026
ROOM_LIGHT_INTENSITY = 600.0


def build_room(room_path: Path, *, rebuild: bool = False) -> Path:
    """Construye room.usd; requiere pxr/Isaac y nunca se ejecuta al importar."""
    if room_path.exists() and not rebuild:
        return room_path
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
    from PIL import Image, ImageDraw

    room_path.parent.mkdir(parents=True, exist_ok=True)
    texture_path = room_path.parent / "textures" / "room.png"
    if not texture_path.exists():
        texture_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (TEXTURE_SIZE, TEXTURE_SIZE), (140, 148, 155))
        draw = ImageDraw.Draw(image)
        step = max(1, TEXTURE_SIZE // 16)
        for y in range(0, TEXTURE_SIZE, step):
            for x in range(0, TEXTURE_SIZE, step):
                if ((x // step) + (y // step) + TEXTURE_SEED) % 2:
                    draw.rectangle((x, y, x + step, y + step), fill=(156, 162, 166))
        image.save(texture_path)
    stage = Usd.Stage.CreateNew(str(room_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    room = UsdGeom.Xform.Define(stage, "/Room")
    materials = UsdGeom.Scope.Define(stage, "/Room/Materials")

    def material(name: str, color: tuple[float, float, float]):
        mat = UsdShade.Material.Define(stage, f"{materials.GetPath()}/{name}")
        shader = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        uv = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/UV")
        uv.CreateIdAttr("UsdPrimvarReader_float2")
        uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        texture = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("textures/room.png")
        )
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            uv.ConnectableAPI(), "result"
        )
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture.ConnectableAPI(), "rgb"
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    half_w, half_d = ROOM_WIDTH_M / 2, ROOM_DEPTH_M / 2
    panels = {
        "Floor": [(-half_w, -half_d, FLOOR_Z_M), (half_w, -half_d, FLOOR_Z_M),
                  (half_w, half_d, FLOOR_Z_M), (-half_w, half_d, FLOOR_Z_M)],
        "North": [(-half_w, half_d, FLOOR_Z_M), (half_w, half_d, FLOOR_Z_M),
                  (half_w, half_d, ROOM_HEIGHT_M), (-half_w, half_d, ROOM_HEIGHT_M)],
        "South": [(half_w, -half_d, FLOOR_Z_M), (-half_w, -half_d, FLOOR_Z_M),
                  (-half_w, -half_d, ROOM_HEIGHT_M), (half_w, -half_d, ROOM_HEIGHT_M)],
        "East": [(half_w, half_d, FLOOR_Z_M), (half_w, -half_d, FLOOR_Z_M),
                 (half_w, -half_d, ROOM_HEIGHT_M), (half_w, half_d, ROOM_HEIGHT_M)],
        "West": [(-half_w, -half_d, FLOOR_Z_M), (-half_w, half_d, FLOOR_Z_M),
                 (-half_w, half_d, ROOM_HEIGHT_M), (-half_w, -half_d, ROOM_HEIGHT_M)],
    }
    room_mat = material("RoomSurface", (0.55, 0.58, 0.62))
    for name, points in panels.items():
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
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(room_mat)

    for index, x in enumerate((-1.5, 1.5)):
        light = UsdLux.RectLight.Define(stage, f"/Lighting/RoomKey_{index}")
        light.CreateIntensityAttr(ROOM_LIGHT_INTENSITY)
        light.CreateWidthAttr(1.5)
        light.CreateHeightAttr(1.5)
        UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(x, 0.0, ROOM_HEIGHT_M - 0.1)
        )
    stage.SetDefaultPrim(room.GetPrim())
    stage.GetRootLayer().Save()
    return room_path
