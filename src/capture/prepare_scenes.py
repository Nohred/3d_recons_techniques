"""Prepara escenas canónicas por objeto, incluido su cuarto dimensionado."""
from pathlib import Path
import traceback
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.paths import object_asset, scene_path, prepare_original, select_object_ids
from common.scene_utils import create_scene

# ================= CONFIGURACION EDITABLE =================
OBJETOS = None  # None: todos los USDZ; o, por ejemplo, ["mug_with_coffee"]
REBUILD_SCENES = True
MOSTRAR_VENTANA = False
# ==========================================================


def main() -> None:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": not MOSTRAR_VENTANA})
    try:
        for object_id in select_object_ids(OBJETOS):
            asset = prepare_original(object_id)
            destination = scene_path(object_id)
            if destination.exists() and not REBUILD_SCENES:
                print(f"OMITIDO (escena existente): {destination}")
                continue
            from pxr import Usd
            destination.parent.mkdir(parents=True, exist_ok=True)
            stage = Usd.Stage.CreateNew(str(destination))
            info = create_scene(stage, asset)
            stage.GetRootLayer().Save()
            print(f"Escena preparada: {destination}; "
                  f"pared={info['room']['wall_distance_m']:.3f} m; "
                  f"textura_sha256={info['room']['texture_sha256']}")
    finally:
        app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
