# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Synthetic-data 3D reconstruction benchmark. For each USDZ in `3D_Models/`, Isaac Sim builds a scene and
captures an orbit video, 3 fixed RGBD views and a ChArUco calibration render; the object is then
reconstructed with four techniques (`nerfacto`, `splatfacto`, `photogrammetry`, `rgbd_fusion`), cleaned,
registered against a Ground Truth point cloud and scored. Code, comments and logs are in Spanish.

## Commands

Single entry point: `./run.sh <stage> [object ...]` (`./run.sh help` lists stages and env vars).

```bash
./run.sh status [--summary]            # per-object progress table (read-only, fast)
DRY_RUN=1 ./run.sh all sneaker         # print what would run
./run.sh rgbd-fusion sneaker           # seconds
./run.sh clean sneaker                 # seconds; METHODS=splatfacto,... to restrict
./run.sh evaluate                      # needs ground_truth.ply (./run.sh gt, Isaac)
FORCE=1 ./run.sh <stage> obj           # redo existing outputs (default: skip)
```

There is no test suite or linter. Cheap checks: `bash -n run.sh src/reconstruct/*.sh`,
`python3 -m py_compile <files>`, `DRY_RUN=1 ./run.sh ...`, and `~/isaacsim/python.sh <isaac_script> --help`
(argparse exits before `SimulationApp` starts).

**Do not launch capture stages (`prepare rgbd orbit calibration gt capture all`) or reconstruction
stages (`process nerf splat recon photogrammetry`) unless the user explicitly asks** — they take
minutes to hours per object and occupy the GPU. Data for 100 objects already exists in `data/`.

## Three interpreters

| Interpreter | Used for | Scripts |
|---|---|---|
| `~/isaacsim/python.sh` (`ISAAC_PYTHON`, py3.12) | anything importing `isaacsim`/`pxr`/`omni` | `src/capture/*.py`, `src/ground_truth/usdz_to_plt.py`, `src/reconstruct/cameras/generate_calibration_views.py` |
| `pixi run` inside `nerfstudio/` | `ns-process-data`, `ns-train`, `ns-export`, `splat_centers.py` | called via `ns()` in `src/reconstruct/lib.sh` |
| system `python3` (miniconda, open3d + cv2 + plyfile) | point-cloud work | `rgbd_fusion.py`, `src/evaluation/*`, `scripts/*` |

COLMAP for dense photogrammetry lives in the conda env `colmap_cuda`; `run.sh` prepends `COLMAP_BIN_DIR` to
`PATH` only for that stage (don't put that env on PATH globally — it shadows `python3`).

## Architecture

- **Everything is keyed by `object_id`** = USDZ stem; all artifacts live under `data/<object_id>/`
  (layout in README). Path helpers: `src/capture/common/paths.py`. Technique → export/clean file paths and
  cleaning params: `src/evaluation/methods.py` (single registry used by clean, evaluate and `scripts/status.py`;
  add a new technique there).
- **Python scripts** keep an `# ==== CONFIGURACION EDITABLE ====` block of module constants (defaults);
  `src/cli_utils.parse_cli()` lets the CLI override only `OBJETOS` and the overwrite flag (`--overwrite`), via
  `global` reassignment at the top of `main()`. Scripts insert the repo root into `sys.path` and import
  as `src.<pkg>.<module>` (capture scripts also resolve `common.*` from their own dir).
- **Bash reconstruction** (`src/reconstruct/lib.sh`, sourced by `recon_*.sh`): two phases — COLMAP for all
  objects first (failures recorded, not fatal), then train/export only those that passed. Idempotency via
  marker files `reconstruction/.process.done`, `.train_<method>.done` (contains the `config.yml` path),
  `.export_<method>.done`. Redoing `process` deletes downstream markers. Every stage appends to
  `reconstruction/stages.csv` and `reports/reconstruction_summary.csv` (`record()`; `rgbd_fusion.py`
  mimics the same format). A `flock` on `.recon.lock` prevents concurrent GPU runs.
- **Overwrite policy everywhere**: existing complete outputs are skipped; incomplete ones raise unless
  overwrite is set. Never write over `data/<obj>/assets/original.usdz` with a different file.
- **Frames**: Isaac world is Z-up, meters. RGBD depth is `float32` z-distance to the image plane, `0` = invalid.
  `rgbd_fusion` works in the ChArUco board frame (OpenCV convention, +Z into the floor, so the object has z<0).
  nerfstudio/COLMAP outputs are in arbitrary frames/scales — evaluation always registers with scale
  (`src/evaluation/registration.py`: FPFH+RANSAC then scaled ICP).
- `simulation_private/` holds simulator truth (board/camera poses). It must not be used as reconstruction
  input — only for diagnostics/evaluation.
- Camera parameters (`ANCHO/ALTO, AZIMUTS, ELEVACION, FOV_HORIZONTAL, MARGEN`) are duplicated between
  `generate_rgbd_three_cameras.py` and `generate_calibration_views.py` and must stay identical.

## Known state

- No `ground_truth.ply` has been generated yet, so `evaluate` has never produced results.
- Nerfacto/photogrammetry cleaning uses generic Splatfacto-derived params (needs tuning). Some objects have a
  `point_cloud_clean.ply` from an earlier out-of-repo nerf cleaner; `clean` skips them unless `FORCE=1`.
- `REALSENSE_D455` depth mode is not integrated; only `IDEAL` works.
- `nerfstudio/` is a separate checkout (gitignored), not project code.
