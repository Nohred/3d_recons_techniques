# Flujo de captura

El flujo activo prepara una escena por objeto y después ejecuta dos
capturadores sobre esa misma escena:

```text
data/<object_id>/
├── assets/original.usdz
├── scene/scene.usd
└── raw/
    ├── orbit.mp4
    └── rgbd/camera_00/{rgb.png,depth.npy}
       rgbd/camera_01/{rgb.png,depth.npy}
       rgbd/camera_02/{rgb.png,depth.npy}
assets/environment/
├── room.usd                 # legado; ya no se referencia en escenas nuevas
└── textures/generated_room.png
```

`src/capture/generate_orbit_room.py` es el capturador RGB oficial.
`src/capture/legacy/generate_orbit.py` se conserva solamente como histórico.
`generate_rgbd_three_cameras.py` produce tres vistas fijas a 120 grados.
`prepare_scenes.py` genera el cuarto dentro de cada `scene.usd`, dimensionado
por objeto; no guarda cámaras de captura persistentes.
Si un USDZ trae una cámara propia, se desactiva bajo `/Object`; una cámara
persistida fuera del objeto se considera una escena incompatible.
La cámara `/OmniverseKit_Persp` que Isaac Sim crea para su viewport se
desactiva y no se considera parte de la escena base.

## Preparación

Ejecutar con el `python.sh` de Isaac Sim desde cualquier directorio:

```bash
cd /home/ccdatos/Escritorio/3D_reconstruction
/ruta/a/isaac-sim/python.sh src/capture/prepare_scenes.py
```

Las variables editables están al principio de cada script. `OBJETOS = None`
descubre los USDZ bajo `3D_Models`; también se puede elegir, por ejemplo,
`OBJETOS = ["mug_with_coffee"]`. La preparación copia cada modelo una sola
vez a `data/<object_id>/assets/original.usdz` y rechaza sobrescribir un
original diferente. `REBUILD_ROOM` y `REBUILD_SCENES` deben activarse
explícitamente para reconstruir recursos; las escenas anteriores no se
reutilizan silenciosamente.

El cuarto mide 6 x 6 x 4 metros, tiene suelo en Z=0, cuatro paredes,
textura procedural reproducible e iluminación fija dentro de `/Room`
(una dome light y dos rect lights). El objeto se centra
horizontalmente y se apoya en ese suelo. Si no cabe, la preparación o la
captura termina con un error; no se escala ni se mueve el cuarto.

La textura usada por el cuarto es la única imagen procedural
`assets/environment/textures/generated_room.png` (2048×2048). Se aplica al
suelo y a las cuatro paredes con transformaciones UV distintas y se registra
su SHA-256 como `capture:textureSha256` en `/Room`. El cuarto usa paredes a
`1.25 × distancia_de_órbita`, suelo justo debajo de la base del objeto, sin
mesa ni techo, y conserva la iluminación legacy (Dome 400 y dos Rect 600,
`normalize=False`).

Después de preparar una escena:

```bash
/ruta/a/isaac-sim/python.sh src/capture/generate_rgbd_three_cameras.py
/ruta/a/isaac-sim/python.sh src/capture/generate_orbit_room.py
```

La órbita oficial usa 8 segundos a 30 FPS (240 frames), una sola vuelta,
elevación lineal de 10° a 65° y FOV horizontal de 50°. El script imprime el
paso azimutal nominal de 300 frames extraídos y avisa si supera 3°.

Ambos comandos omiten resultados completos. RGBD no reemplaza resultados
incompletos salvo que `SOBRESCRIBIR_RGBD = True`; la órbita aplica la misma
política con `SOBRESCRIBIR_VIDEO = True`. No se escriben los protocolos
antiguos `data/raw/<protocolo>/...`, reportes de cámara ni `cameras.json`.

## Profundidad

El modo predeterminado es `IDEAL`: `depth.npy` es `float32`, forma
`(ALTO, ANCHO)`, metros de distancia al plano de imagen, y `0.0` significa
inválido/no observado. No se normaliza por imagen. `REALSENSE_D455` es una
opción explícita separada y nunca es fallback automático. Requiere que la
instalación activa exponga `SingleViewDepthSensorAsset` y
`DepthSensorDistance`, además del asset `rsd455.usd`; el script falla si la
salida registrada RGB-depth no está verificada. La simplificación documentada
para el modo ideal es pinhole, RGB co-localizado y profundidad perfecta, no un
sensor RealSense.

El código fue escrito contra las APIs de Isaac Sim 5.1 mencionadas por los
scripts originales. Este entorno de trabajo no contiene `python.sh` ni una
instalación Isaac Sim ejecutable, por lo que la versión efectiva y la firma
exacta del backend RealSense deben comprobarse en la instalación local antes
de activar ese modo.

## Inventario, sin limpieza destructiva

* **Conservar:** `3D_Models/`, `data/raw/orbita/` y
  `data/raw/orbita_room_test_v1/`; contienen USDZ y videos históricos que no
  se sobrescriben.
* **Archivar posteriormente:** `simulation_private/orbita/` y
  `simulation_private/orbita_room_test_v1/`; contienen reportes, previews,
  texturas y escenas antiguas por objeto. Son regenerables, pero sirven para
  reproducir resultados anteriores.
* **Migrar explícitamente:** USDZ seleccionados hacia
  `data/<object_id>/assets/` y escenas nuevas hacia
  `data/<object_id>/scene/`; la preparación es idempotente y no borra los
  originales.
* **Candidatos a eliminar con aprobación:** `data/_work/` y sus frames
  temporales, previews y cachés `__pycache__`. Son regenerables, pero los
  primeros pueden ser evidencia de ejecuciones antiguas.
* **No marcar como basura:** `metadata/limites_objetos.json`, scripts de
  evaluación y resultados de reconstrucción; pueden ser entradas de otros
  flujos.

## Ground Truth y evaluación

Generar el Ground Truth por cada objeto canónico:

```bash
/ruta/a/isaac-sim/python.sh src/ground_truth/usdz_to_plt.py
```

Cada resultado se escribe en `data/<object_id>/ground_truth/ground_truth.ply`.
El script usa variables editables (`OBJETOS`, `PUNTOS_DESEADOS` y
`SOBRESCRIBIR`) y no reemplaza un GT existente por defecto.

Limpiar los exports de Splatfacto:

```bash
python src/evaluation/clean_reconstructions.py
```

Solo se procesa
`reconstruction/exports/splatfacto/gaussian_centers_NOT_SURFACE.ply` y se
genera `gaussian_centers_clean.ply` en la misma carpeta. Nerfacto queda
preparado en el script, pero deshabilitado hasta que exista su limpiador
específico.

Registrar y evaluar:

```bash
python src/evaluation/run_evaluation.py
```

Se generan `reports/<object_id>.json` y `reports/summary.json`. El flujo
registra traslación, rotación y escala antes de calcular Chamfer, RMSE,
Hausdorff, precisión, recall, F-Score y PSNR geométrico.
