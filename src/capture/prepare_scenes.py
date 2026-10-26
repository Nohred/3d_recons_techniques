"""Prepara una vez el cuarto compartido y las escenas canónicas por objeto."""
from pathlib import Path
import traceback
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.paths import (
    ROOM_USD,
    object_asset,
    scene_path,
    prepare_original,
    select_object_ids,
)
from src.common.room import build_room

# ================= CONFIGURACION EDITABLE =================
OBJETOS = None  # None: todos los USDZ; o, por ejemplo, ["mug_with_coffee"]
REBUILD_ROOM = False
REBUILD_SCENES = False
MOSTRAR_VENTANA = False
# ==========================================================


def main() -> None:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": not MOSTRAR_VENTANA})
    try:
        texture = ROOM_USD.parent / "textures" / "room.png"
        if ROOM_USD.exists() and not texture.exists() and not REBUILD_ROOM:
            raise RuntimeError(f"Falta la textura del cuarto compartido: {texture}")
        if ROOM_USD.exists() and not REBUILD_ROOM:
            print(f"Reutilizando cuarto existente: {ROOM_USD}")
        else:
            build_room(ROOM_USD, rebuild=REBUILD_ROOM)
            print(f"Cuarto preparado: {ROOM_USD}")
        for object_id in select_object_ids(OBJETOS):
            asset = prepare_original(object_id)
            destination = scene_path(object_id)
            if destination.exists() and not REBUILD_SCENES:
                print(f"OMITIDO (escena existente): {destination}")
                continue
            from pxr import Usd
            from src.common.scene_utils import create_scene

            destination.parent.mkdir(parents=True, exist_ok=True)
            stage = Usd.Stage.CreateNew(str(destination))
            create_scene(stage, asset, ROOM_USD)
            stage.GetRootLayer().Save()
            print(f"Escena preparada: {destination}")
    finally:
        app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
