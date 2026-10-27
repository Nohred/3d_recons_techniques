"""Tablero ChArUco para calibrar las 3 camaras RGB-D (geometria, imagen y verificacion).

Este modulo NO depende de Isaac Sim (solo numpy, Pillow y OpenCV >= 4.7).
Generar la imagen del tablero:   python charuco_board.py
"""
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np

# ================= CONFIGURACION EDITABLE =================
ROOT = Path(__file__).resolve().parents[3]
CARPETA_TABLERO = ROOT / "assets" / "calibration"
NOMBRE_TABLERO = "charuco_7x5_4x4_50"

CASILLAS_X = 7                  # columnas de casillas
CASILLAS_Y = 5                  # filas de casillas
DICCIONARIO = "DICT_4X4_50"     # marcadores de 4x4 bits: mas robustos a baja resolucion
RELACION_MARCADOR = 0.70        # lado del marcador / lado de la casilla
PX_POR_CASILLA = 200            # resolucion de la textura
MARGEN_CASILLAS = 0.5           # borde blanco alrededor, en casillas
NIVEL_BLANCO = 235              # evita saturar el blanco bajo la iluminacion de la escena
NIVEL_NEGRO = 25
# ==========================================================


def _cv2():
    import cv2
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
        raise RuntimeError("Se requiere OpenCV >= 4.7 con cv2.aruco.CharucoDetector "
                           "(pip install opencv-contrib-python).")
    return cv2


def margen_px() -> int:
    return int(round(MARGEN_CASILLAS * PX_POR_CASILLA))


def tamano_imagen_px() -> tuple:
    m = margen_px()
    return CASILLAS_X * PX_POR_CASILLA + 2 * m, CASILLAS_Y * PX_POR_CASILLA + 2 * m


def crear_tablero(longitud_casilla: float = 1.0):
    """Devuelve (cv2.aruco.CharucoBoard, diccionario). Las unidades las da longitud_casilla."""
    cv2 = _cv2()
    diccionario = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICCIONARIO))
    tablero = cv2.aruco.CharucoBoard((CASILLAS_X, CASILLAS_Y), float(longitud_casilla),
                                     float(longitud_casilla) * RELACION_MARCADOR, diccionario)
    return tablero, diccionario


def generar_imagen() -> np.ndarray:
    """Imagen en escala de grises (uint8) con el tablero y su margen blanco."""
    tablero, _ = crear_tablero(1.0)
    ancho, alto = tamano_imagen_px()
    imagen = tablero.generateImage((ancho, alto), marginSize=margen_px(), borderBits=1)
    nivel = NIVEL_NEGRO + (imagen.astype(np.float32) / 255.0) * (NIVEL_BLANCO - NIVEL_NEGRO)
    return np.clip(np.rint(nivel), 0, 255).astype(np.uint8)


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def especificacion(longitud_casilla_m: float | None = None) -> dict:
    """Datos del tablero que un usuario real conoce (entrada legitima de la calibracion)."""
    ancho, alto = tamano_imagen_px()
    datos = {
        "tipo": "ChArUco", "casillas_x": CASILLAS_X, "casillas_y": CASILLAS_Y,
        "diccionario": DICCIONARIO, "relacion_marcador": RELACION_MARCADOR,
        "imagen_px": [ancho, alto], "px_por_casilla": PX_POR_CASILLA, "margen_px": margen_px(),
        "marco_tablero": "origen = esquina superior izquierda del area de casillas, "
                         "X a la derecha, Y hacia abajo en la imagen, Z = X x Y (hacia dentro del tablero)",
    }
    if longitud_casilla_m is not None:
        datos["longitud_casilla_m"] = float(longitud_casilla_m)
        datos["longitud_marcador_m"] = float(longitud_casilla_m) * RELACION_MARCADOR
    return datos


def asegurar_imagen(sobrescribir: bool = False) -> Path:
    """Crea (si no existe) assets/calibration/<nombre>.png y su JSON; devuelve la ruta del PNG."""
    from PIL import Image
    CARPETA_TABLERO.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_TABLERO / f"{NOMBRE_TABLERO}.png"
    if sobrescribir or not ruta.is_file():
        Image.fromarray(generar_imagen(), mode="L").convert("RGB").save(ruta)
    datos = especificacion()
    datos["archivo"] = ruta.name
    datos["sha256"] = sha256_archivo(ruta)
    (CARPETA_TABLERO / f"{NOMBRE_TABLERO}.json").write_text(
        json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta


# ---------------------------------------------------------------- geometria (marco mundo, Z arriba)
def marco_tablero_en_mundo(centro_xy, z: float, longitud_casilla_m: float) -> np.ndarray:
    """T_mundo<-tablero (4x4). Imagen vista desde arriba: derecha=+X mundo, arriba=+Y mundo.

    Marco OpenCV del tablero: X derecha, Y hacia abajo en la imagen, Z = X x Y (hacia el suelo).
    El origen es la esquina superior izquierda del area de casillas (sin margen)."""
    ancho = CASILLAS_X * longitud_casilla_m
    alto = CASILLAS_Y * longitud_casilla_m
    T = np.eye(4)
    T[:3, 0] = (1.0, 0.0, 0.0)
    T[:3, 1] = (0.0, -1.0, 0.0)
    T[:3, 2] = (0.0, 0.0, -1.0)
    T[:3, 3] = (centro_xy[0] - ancho / 2.0, centro_xy[1] + alto / 2.0, z)
    return T


def esquinas_plano_mundo(centro_xy, z: float, longitud_casilla_m: float) -> np.ndarray:
    """Cuatro esquinas del plano texturizado (incluye margen), antihorario visto desde arriba:
    abajo-izq, abajo-der, arriba-der, arriba-izq."""
    margen = MARGEN_CASILLAS * longitud_casilla_m
    mx = CASILLAS_X * longitud_casilla_m / 2.0 + margen
    my = CASILLAS_Y * longitud_casilla_m / 2.0 + margen
    cx, cy = float(centro_xy[0]), float(centro_xy[1])
    return np.array([[cx - mx, cy - my, z], [cx + mx, cy - my, z],
                     [cx + mx, cy + my, z], [cx - mx, cy + my, z]], dtype=float)


def proyectar_usd(puntos_mundo: np.ndarray, c2w: np.ndarray, K: np.ndarray):
    """Proyecta puntos con una camara USD (mira a -Z, +Y arriba). Devuelve (u, v, z_cv)."""
    w2c = np.linalg.inv(c2w)
    p = (w2c @ np.c_[puntos_mundo, np.ones(len(puntos_mundo))].T).T[:, :3]
    x, y, z = p[:, 0], -p[:, 1], -p[:, 2]  # convencion OpenCV
    with np.errstate(divide="ignore", invalid="ignore"):
        return K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2], z


def metricas_vista(centro_xy, z: float, longitud_casilla_m: float,
                   c2w: np.ndarray, K: np.ndarray, ancho: int, alto: int) -> dict:
    """Comprueba que el area de casillas cabe en la imagen y estima su tamano en pixeles."""
    esq = esquinas_plano_mundo(centro_xy, z, longitud_casilla_m)
    u, v, zc = proyectar_usd(esq, c2w, K)
    ok = bool(np.all(zc > 0))
    dentro = ok and bool(np.all((u >= 0) & (u < ancho) & (v >= 0) & (v < alto)))
    holgura = float(min(u.min(), v.min(), ancho - 1 - u.max(), alto - 1 - v.max())) if ok else float("-inf")
    c = np.array([[centro_xy[0], centro_xy[1], z]], dtype=float)
    pts = np.vstack([c, c + [longitud_casilla_m, 0, 0], c + [0, longitud_casilla_m, 0]])
    uu, vv, zz = proyectar_usd(pts, c2w, K)
    px_x = float(math.hypot(uu[1] - uu[0], vv[1] - vv[0])) if np.all(zz > 0) else 0.0
    px_y = float(math.hypot(uu[2] - uu[0], vv[2] - vv[0])) if np.all(zz > 0) else 0.0
    pos = c2w[:3, 3]
    hacia_camara = pos - c[0]
    cos = float(hacia_camara[2] / np.linalg.norm(hacia_camara))
    return {"dentro_de_imagen": dentro, "holgura_px": holgura,
            "px_casilla_min": min(px_x, px_y), "px_casilla_x": px_x, "px_casilla_y": px_y,
            "angulo_con_normal_deg": float(math.degrees(math.acos(max(-1.0, min(1.0, cos)))))}


def elegir_longitud_casilla(centro_xy, z: float, poses_c2w: list, K: np.ndarray, ancho: int, alto: int,
                            ancho_plano_max_m: float, holgura_min_px: float = 30.0,
                            reduccion: float = 0.97, ancho_plano_min_m: float = 0.0) -> tuple:
    """Mayor tablero (hasta ancho_plano_max_m) cuyo plano completo cabe en TODAS las camaras.

    El ancho del plano incluye el margen blanco. Devuelve (longitud_casilla_m, metricas_por_camara)
    o lanza RuntimeError si no cabe con al menos ancho_plano_min_m."""
    divisor = CASILLAS_X + 2.0 * MARGEN_CASILLAS
    ancho_plano = float(ancho_plano_max_m)
    while ancho_plano >= max(ancho_plano_min_m, 1e-6):
        sq = ancho_plano / divisor
        mets = [metricas_vista(centro_xy, z, sq, c2w, K, ancho, alto) for c2w in poses_c2w]
        if all(m["dentro_de_imagen"] and m["holgura_px"] >= holgura_min_px for m in mets):
            return sq, mets
        ancho_plano *= reduccion
    raise RuntimeError("El tablero no cabe completo en las tres camaras con la holgura pedida; "
                       "revisa elevacion, FOV o la altura del objeto.")


# ---------------------------------------------------------------- deteccion (verificacion)
def detectar(imagen_rgb: np.ndarray) -> dict:
    """Cuenta marcadores y esquinas ChArUco detectadas (solo verificacion de visibilidad)."""
    cv2 = _cv2()
    tablero, _ = crear_tablero(1.0)
    detector = cv2.aruco.CharucoDetector(tablero)
    gris = cv2.cvtColor(np.ascontiguousarray(imagen_rgb[:, :, :3]), cv2.COLOR_RGB2GRAY)
    esquinas, ids, marcadores, ids_marc = detector.detectBoard(gris)
    return {"marcadores": 0 if ids_marc is None else int(len(ids_marc)),
            "esquinas": 0 if ids is None else int(len(ids)),
            "esquinas_totales": (CASILLAS_X - 1) * (CASILLAS_Y - 1)}


if __name__ == "__main__":
    ruta = asegurar_imagen(sobrescribir="--force" in sys.argv)
    ancho, alto = tamano_imagen_px()
    print(f"Tablero: {ruta} ({ancho}x{alto} px)")
    print(f"JSON:    {ruta.with_suffix('.json')}")
