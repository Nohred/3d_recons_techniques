from __future__ import annotations

from pathlib import Path
import shutil


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODELS_ROOT = REPOSITORY_ROOT / "3D_Models"
DATA_ROOT = REPOSITORY_ROOT / "data"
SHARED_ASSETS_ROOT = REPOSITORY_ROOT / "assets"
ROOM_USD = SHARED_ASSETS_ROOT / "environment" / "room.usd"


def object_root(object_id: str) -> Path:
    return DATA_ROOT / object_id


def object_asset(object_id: str) -> Path:
    return object_root(object_id) / "assets" / "original.usdz"


def scene_path(object_id: str) -> Path:
    return object_root(object_id) / "scene" / "scene.usd"


def orbit_output(object_id: str) -> Path:
    return object_root(object_id) / "raw" / "orbit.mp4"


def rgbd_output(object_id: str, index: int) -> Path:
    return object_root(object_id) / "raw" / "rgbd" / f"camera_{index:02d}"


#def discover_model_files() -> list[Path]:
#    if not MODELS_ROOT.is_dir():
#        raise RuntimeError(f"No existe la carpeta de modelos: {MODELS_ROOT}")
#    return sorted(
#        path for path in MODELS_ROOT.rglob("*.usdz") if path.is_file()
#    )

# Debug files detection

def discover_model_files() -> list[Path]:
    if not MODELS_ROOT.is_dir():
        raise RuntimeError(f"No existe la carpeta de modelos: {MODELS_ROOT}")
    #path_list = sorted(path for path in MODELS_ROOT.rglob("*.usdz") if path.is_file())
    # Get all .usdz files in the MODELS_ROOT directory without recursion
    path_list = sorted(path for path in MODELS_ROOT.glob("*.usdz") if path.is_file())

    #print(f"DEBUG: discover_model_files found {len(path_list)} .usdz files in {MODELS_ROOT}")
    #print("DEBUG: List of discovered .usdz files:")
    #for path in path_list:  # Print only the first 10 for brevity
    #    print(f"DEBUG: {path}")
    
    return path_list


def discover_object_ids() -> list[str]:
    return sorted(path.stem for path in discover_model_files())


def resolve_model(object_id: str) -> Path:
    matches = [path for path in discover_model_files() if path.stem == object_id]
    if not matches:
        raise FileNotFoundError(
            f"No hay USDZ para object_id={object_id!r} bajo {MODELS_ROOT}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Colisión de object_id={object_id!r}; candidatos: {matches}"
        )
    return matches[0]


def prepare_original(object_id: str, source: Path | None = None) -> Path:
    """Copia un modelo al convenio canónico sin sobrescribir otro original."""
    source = (source or resolve_model(object_id)).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    target = object_asset(object_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != source.read_bytes():
            raise FileExistsError(
                f"{target} ya contiene un original distinto; no se sobrescribe."
            )
        return target
    shutil.copy2(source, target)
    return target


def select_object_ids(selected: list[str] | tuple[str, ...] | None) -> list[str]:
    ids = discover_object_ids() if selected is None else list(selected)
    if len(set(ids)) != len(ids):
        raise ValueError(f"OBJETOS contiene identificadores repetidos: {ids}")
    for object_id in ids:
        if not object_id or Path(object_id).name != object_id:
            raise ValueError(f"object_id inválido: {object_id!r}")
        resolve_model(object_id)
    return ids
