# Reconstrucción 3D sintética: captura automatizada + comparación de técnicas

Pipeline que, para cada modelo USDZ de `3D_Models/`, genera una escena en Isaac Sim,
captura datos sintéticos (video orbital, 3 vistas RGBD y calibración ChArUco), reconstruye
el objeto con cuatro técnicas y las evalúa contra el Ground Truth del modelo.

| Técnica          | Entrada              | Herramienta                              | Export                                   |
|------------------|----------------------|------------------------------------------|------------------------------------------|
| `nerfacto`       | `raw/orbit.mp4`      | nerfstudio (`ns-train nerfacto`)         | `exports/nerfacto/point_cloud.ply`       |
| `splatfacto`     | `raw/orbit.mp4`      | nerfstudio (`ns-train splatfacto`)       | `exports/splatfacto/gaussian_centers_NOT_SURFACE.ply` |
| `photogrammetry` | COLMAP de la órbita  | COLMAP denso (patch-match + fusion)      | `exports/photogrammetry/fused.ply`       |
| `rgbd_fusion`    | `raw/rgbd/` + `raw/calibration/` | ChArUco + retroproyección (Open3D) | `exports/rgbd_fusion/fused.ply`      |

## Requisitos

- **Isaac Sim 5.x** (`~/isaacsim/python.sh`, o `ISAAC_PYTHON=...`): escenas, capturas, calibración y GT.
- **nerfstudio** clonado en `nerfstudio/` con su entorno **pixi** (`pixi run ns-*`); gcc/g++-11 para compilar gsplat.
- **COLMAP** con CUDA para fotogrametría (en `PATH` o en `COLMAP_BIN_DIR`, por defecto `~/miniconda3/envs/colmap_cuda/bin`).
- **ffmpeg/ffprobe** (codificación de la órbita).
- **python3** con `numpy open3d opencv-contrib-python plyfile pillow matplotlib` (fusión RGBD, limpieza, evaluación).

## Uso rápido

Todo pasa por `./run.sh <etapa> [objeto ...]`. Sin objetos se procesan todos los disponibles.
Cada etapa **omite lo que ya existe**; `FORCE=1` lo rehace. `DRY_RUN=1` solo muestra qué haría.

```bash
./run.sh help                      # etapas y variables
./run.sh status                    # tabla de avance por objeto (status --summary: totales)

./run.sh all                       # pipeline completo para todos los objetos
./run.sh all sneaker beach_ball    # ... o solo para algunos

# Por partes
./run.sh capture sneaker           # prepare + rgbd + orbit + calibration (Isaac Sim)
./run.sh recon sneaker             # COLMAP + Nerfacto + Splatfacto (nerfstudio)
./run.sh photogrammetry sneaker    # COLMAP denso (requiere recon/process previo)
./run.sh rgbd-fusion sneaker       # fusión de las 3 vistas RGBD (segundos)
./run.sh post sneaker              # gt + clean + evaluate

# Opciones
FORCE=1 ./run.sh orbit sneaker                      # regrabar la órbita
METHODS=splatfacto,rgbd_fusion ./run.sh evaluate    # evaluar solo algunas técnicas
NERF_ITERS=30000 SPLAT_ITERS=30000 ./run.sh recon   # más iteraciones
MAX_OBJECTS=0 ./run.sh photogrammetry               # sin límite (por defecto 10 objetos)
```

Las etapas de captura y reconstrucción son largas (minutos a horas por objeto); `rgbd-fusion`,
`clean` y `evaluate` tardan segundos. Un fallo en un objeto no detiene el lote: se registra y
se continúa. Las etapas compuestas (`capture`, `post`, `all`) continúan con la siguiente etapa
y reportan al final cuáles fallaron.

### Etapas

| Etapa            | Script                                                  | Salida (bajo `data/<obj>/`)                    |
|------------------|---------------------------------------------------------|------------------------------------------------|
| `prepare`        | `src/capture/prepare_scenes.py`                         | `assets/original.usdz`, `scene/scene.usd`      |
| `rgbd`           | `src/capture/generate_rgbd_three_cameras.py`            | `raw/rgbd/camera_0{0,1,2}/{rgb.png,depth.npy}` |
| `orbit`          | `src/capture/generate_orbit_room.py`                    | `raw/orbit.mp4`, `raw/orbit_report.json`       |
| `calibration`    | `src/reconstruct/cameras/generate_calibration_views.py` | `raw/calibration/`, `simulation_private/calibration_ground_truth.json` |
| `gt`             | `src/ground_truth/usdz_to_plt.py`                       | `ground_truth/ground_truth.ply`                |
| `process`        | `src/reconstruct/recon_process.sh`                      | `reconstruction/processed/` (frames + COLMAP)  |
| `nerf`/`splat`/`recon` | `src/reconstruct/recon_{nerf,splat,all}.sh`       | `reconstruction/runs/`, `reconstruction/exports/{nerfacto,splatfacto}/` |
| `photogrammetry` | `src/reconstruct/recon_photogrammetry.sh`               | `reconstruction/exports/photogrammetry/`       |
| `rgbd-fusion`    | `src/reconstruct/rgbd_fusion.py`                        | `reconstruction/exports/rgbd_fusion/{fused.ply,poses.json}` |
| `clean`          | `src/evaluation/clean_reconstructions.py`               | `*_clean.ply` junto a cada export              |
| `evaluate`       | `src/evaluation/run_evaluation.py`                      | `reports/evaluation/<obj>__<metodo>.json`, `summary.{json,csv}` |

Los scripts Python también se pueden ejecutar directamente: aceptan `[objeto ...] [--overwrite]`
(y `--methods a,b` los de limpieza/evaluación). Sus parámetros finos siguen en el bloque
`CONFIGURACION EDITABLE` al inicio de cada archivo; la CLI solo sobrescribe objetos y sobrescritura.

## Estructura de datos

```text
3D_Models/<obj>.usdz                    # modelos fuente (3D_Models/reserved/ se ignora)
data/<obj>/
├── assets/original.usdz                # copia canónica (nunca se sobrescribe con otro modelo)
├── scene/scene.usd                     # cuarto + luces + objeto
├── raw/
│   ├── orbit.mp4                       # 1280x720, 30 FPS, 8 s
│   ├── rgbd/camera_0{0,1,2}/           # rgb.png + depth.npy
│   └── calibration/                    # ChArUco en el piso, mismas 3 cámaras, sin objeto
├── simulation_private/                 # verdad del simulador: SOLO para diagnóstico/evaluación
├── ground_truth/ground_truth.ply
└── reconstruction/
    ├── processed/                      # frames + transforms.json + colmap/
    ├── runs/                           # entrenamientos nerfstudio
    ├── exports/<metodo>/               # nubes crudas y *_clean.ply
    ├── logs/                           # un log por etapa
    ├── stages.csv                      # historial de etapas del objeto
    └── .process.done, .train_*.done, .export_*.done   # marcadores de etapa
reports/
├── reconstruction_summary.csv          # historial global de etapas
├── colmap_report.csv                   # resultado de COLMAP del último lote
└── evaluation/summary.{csv,json}       # comparación de técnicas
cloud/<obj>/{nerfacto,splatfacto}/      # nubes de 5 objetos versionadas como referencia
```

`data/`, `3D_Models/`, `reports/` y `nerfstudio/` están en `.gitignore`.

## Notas técnicas

**Escena.** `prepare_scenes.py` genera por objeto un cuarto sin techo con paredes a
`1.25 × distancia de órbita`, suelo bajo la base del objeto, textura procedural única
(`assets/environment/textures/generated_room.png`, SHA-256 registrado en `/Room`) e iluminación
fija (Dome 400 + dos Rect 600, `normalize=False`). El objeto se centra y se apoya en el suelo;
si no cabe, la etapa falla (no se escala). Cámaras propias del USDZ se desactivan.

**Órbita.** 240 frames (8 s a 30 FPS), una vuelta, elevación lineal 10°→65°, FOV horizontal 50°.
nerfstudio extrae `NUM_FRAMES=300` imágenes; COLMAP debe registrar ≥ `MIN_REGISTERED_FRACTION`
(0.90) de ellas o no se entrena.

**RGBD.** Tres cámaras a 0/120/240° de azimut y 25° de elevación. Modo `IDEAL` (por defecto):
`depth.npy` es `float32 (720, 1280)`, metros al plano de imagen, `0.0` = inválido; pinhole y RGB
co-localizado. `REALSENSE_D455` es opcional, nunca fallback, y no está verificado en esta instalación.

**Calibración.** El tablero ChArUco 7×5 (`DICT_4X4_50`, `assets/calibration/`) se coloca en el piso
bajo el objeto, que se desactiva solo en la sesión; se renderiza con las mismas cámaras RGBD.
`calibration_inputs.json` contiene lo que conocería un usuario real (K, tablero); la verdad del
simulador queda en `simulation_private/`.

**Fusión RGBD.** Estima la pose de cada cámara respecto al tablero con `solvePnP`, retroproyecta
las tres profundidades al marco del tablero, recorta el suelo (plano del tablero) y una esfera
alrededor del punto donde convergen los ejes ópticos (radio deducido del encuadre). `poses.json`
reporta reproyección y error contra la pose del simulador (≈1 mm en `sneaker`).

**Fotogrametría.** `image_undistorter` → `patch_match_stereo` (consistencia geométrica) →
`stereo_fusion` sobre `processed/colmap/sparse/0`. `MAX_IMAGE_SIZE=1300`. Solo considera objetos
con COLMAP completo; por defecto procesa como máximo `MAX_OBJECTS=10` por ejecución.

**Limpieza y evaluación.** `src/evaluation/clean_pointcloud.py` (recorte esférico alrededor de la
mediana, eliminación del plano del suelo, DBSCAN de la isla mayor, filtro estadístico) con
parámetros por técnica en `src/evaluation/methods.py`. La evaluación registra traslación, rotación
y escala (FPFH + RANSAC + ICP con escala) y calcula Chamfer, RMSE, Hausdorff, precisión, recall,
F-score (umbral 2 % de la diagonal del GT) y PSNR geométrico. Uso manual sobre dos nubes:
`python3 src/evaluation/evaluation.py reconstruccion.ply gt.ply`.

## Utilidades

- `python3 scripts/view_rgbd.py <obj>` — muestra las 3 vistas RGB y su profundidad.
- `python3 src/evaluation/clean_pointcloud.py in.ply [out.ply] --crop-ratio ...` — limpieza manual con parámetros a mano.
- `~/isaacsim/python.sh src/capture/inspect_models.py` — reporte de dimensiones de los USDZ.

## Pendientes conocidos

- Los parámetros de limpieza de `nerfacto` y `photogrammetry` son los genéricos de Splatfacto; falta ajustarlos.
- 4 objetos (`cricket_ball fork iphone old_spoon`) no pasan COLMAP (`reports/failed.txt`).
- El backend RealSense D455 no está integrado.
