from pathlib import Path
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaacsim import SimulationApp


# ---------- VARIABLES EDITABLES ----------

CARPETA_MODELOS = ROOT / "3D_Models"

CARPETA_SALIDA = ROOT / "metadata"

MOSTRAR_VENTANA = False

# ----------------------------------------


app = SimulationApp({
    "headless": not MOSTRAR_VENTANA,
})

from pxr import Gf, Usd, UsdGeom


def medir_objeto(ruta):
    stage = Usd.Stage.Open(str(ruta))

    if stage is None:
        raise RuntimeError("No se pudo abrir el modelo.")

    escala = float(UsdGeom.GetStageMetersPerUnit(stage))
    eje_vertical = str(UsdGeom.GetStageUpAxis(stage))

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
        ],
        useExtentsHint=False,
    )

    caja_total = Gf.Range3d()

    for prim in Usd.PrimRange(
        stage.GetPseudoRoot(),
        Usd.TraverseInstanceProxies(),
    ):
        if prim.IsA(UsdGeom.Mesh):
            caja = cache.ComputeWorldBound(prim).ComputeAlignedBox()

            if not caja.IsEmpty():
                caja_total.UnionWith(caja)

    if caja_total.IsEmpty():
        raise RuntimeError("No se encontró geometría de malla válida.")

    dimensiones = [
        float(valor) * escala
        for valor in caja_total.GetSize()
    ]

    if eje_vertical == "Z":
        ancho, fondo, altura = dimensiones
    elif eje_vertical == "Y":
        ancho, altura, fondo = dimensiones
    else:
        raise RuntimeError(f"Eje vertical no soportado: {eje_vertical}")

    return {
        "archivo": ruta.name,
        "ancho_m": ancho,
        "fondo_m": fondo,
        "altura_m": altura,
        "dimension_minima_m": min(dimensiones),
        "dimension_maxima_m": max(dimensiones),
        "radio_envolvente_m": (
            0.5 * math.sqrt(sum(d * d for d in dimensiones))
        ),
        "meters_per_unit": escala,
        "unidades_explicitas": stage.HasAuthoredMetadata(
            "metersPerUnit"
        ),
        "up_axis_original": eje_vertical,
    }


def obtener_limites(objetos, campo):
    menor = min(objetos, key=lambda objeto: objeto[campo])
    mayor = max(objetos, key=lambda objeto: objeto[campo])

    return {
        "min": menor[campo],
        "objeto_min": menor["archivo"],
        "max": mayor[campo],
        "objeto_max": mayor["archivo"],
    }


try:
    archivos = sorted(
        ruta
        for ruta in CARPETA_MODELOS.iterdir()
        if ruta.suffix.lower() == ".usdz"
    )

    if not archivos:
        raise RuntimeError("La carpeta no contiene archivos USDZ.")

    objetos = []
    errores = []

    for ruta in archivos:
        try:
            objeto = medir_objeto(ruta)
            objetos.append(objeto)

            print(
                f"{ruta.name}: "
                f"ancho={objeto['ancho_m']:.4f} m, "
                f"fondo={objeto['fondo_m']:.4f} m, "
                f"altura={objeto['altura_m']:.4f} m"
            )

            if not objeto["unidades_explicitas"]:
                print("  AVISO: el archivo no declara sus unidades.")

        except Exception as error:
            errores.append({
                "archivo": ruta.name,
                "error": str(error),
            })
            print(f"ERROR: {ruta.name}: {error}")

    if not objetos:
        raise RuntimeError("No se pudo medir ningún objeto.")

    campos = [
        "ancho_m",
        "fondo_m",
        "altura_m",
        "dimension_minima_m",
        "dimension_maxima_m",
        "radio_envolvente_m",
    ]

    limites = {
        campo: obtener_limites(objetos, campo)
        for campo in campos
    }

    print("\n===== LÍMITES DEL CONJUNTO =====")

    for campo, valores in limites.items():
        print(f"\n{campo}")
        print(
            f"  MIN: {valores['min']:.6f} m "
            f"({valores['objeto_min']})"
        )
        print(
            f"  MAX: {valores['max']:.6f} m "
            f"({valores['objeto_max']})"
        )

    reporte = {
        "objetos_procesados": len(objetos),
        "limites": limites,
        "objetos": objetos,
        "errores": errores,
    }

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    destino = CARPETA_SALIDA / "limites_objetos.json"

    destino.write_text(
        json.dumps(reporte, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nReporte guardado: {destino}")
    print(f"Objetos procesados: {len(objetos)}")
    print(f"Errores: {len(errores)}")

finally:
    app.close()