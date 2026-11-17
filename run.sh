#!/usr/bin/env bash
# Orquestador del pipeline: captura (Isaac Sim) -> reconstrucción -> limpieza -> evaluación.
# Uso: ./run.sh <etapa> [objeto ...]      (./run.sh help para la lista)
set -uo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
SRC="$ROOT/src"
RECON="$SRC/reconstruct"

ISAAC_PYTHON="${ISAAC_PYTHON:-$HOME/isaacsim/python.sh}"
PYTHON="${PYTHON:-python3}"
COLMAP_BIN_DIR="${COLMAP_BIN_DIR:-$HOME/miniconda3/envs/colmap_cuda/bin}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
METHODS="${METHODS:-}"
export FORCE DRY_RUN

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Uso: ./run.sh <etapa> [objeto ...]
Sin objetos se procesan todos los disponibles para esa etapa.

Captura (Isaac Sim, ISAAC_PYTHON):
  prepare         3D_Models/<obj>.usdz -> data/<obj>/{assets/original.usdz, scene/scene.usd}
  rgbd            3 vistas RGB + depth            -> data/<obj>/raw/rgbd/camera_0{0,1,2}/
  orbit           video orbital 240 frames        -> data/<obj>/raw/orbit.mp4
  calibration     tablero ChArUco, mismas cámaras -> data/<obj>/raw/calibration/
  capture         prepare + rgbd + orbit + calibration
  gt              Ground Truth                    -> data/<obj>/ground_truth/ground_truth.ply

Reconstrucción:
  process         frames + COLMAP (ns-process-data) -> reconstruction/processed/
  nerf | splat    entrenar + exportar Nerfacto / Splatfacto (incluye process)
  recon           process + nerf + splat
  photogrammetry  COLMAP denso sobre processed/   -> exports/photogrammetry/fused.ply
  rgbd-fusion     fusión de las 3 vistas RGBD     -> exports/rgbd_fusion/fused.ply

Post-proceso:
  clean           limpiar exports de cada técnica -> *_clean.ply
  evaluate        registro + métricas vs GT        -> reports/evaluation/summary.{csv,json}
  post            gt + clean + evaluate

  all             capture + recon + photogrammetry + rgbd-fusion + post
  status          tabla de avance por objeto (status --summary: solo totales)
  help            esta ayuda

Variables de entorno:
  FORCE=1               rehacer salidas existentes (por defecto se omiten)
  DRY_RUN=1             mostrar qué se ejecutaría
  METHODS=a,b           clean/evaluate: splatfacto,nerfacto,photogrammetry,rgbd_fusion
  ISAAC_PYTHON=...      python.sh de Isaac Sim (def. ~/isaacsim/python.sh)
  PYTHON=...            intérprete con open3d/opencv/plyfile (def. python3)
  COLMAP_BIN_DIR=...    carpeta con colmap si no está en PATH (def. ~/miniconda3/envs/colmap_cuda/bin)
  NUM_FRAMES, NERF_ITERS, SPLAT_ITERS, MIN_REGISTERED_FRACTION, SKIP_EXPORT   (ver src/reconstruct/lib.sh)
  MAX_OBJECTS (def. 10; 0 = todos), MAX_IMAGE_SIZE   (ver src/reconstruct/recon_photogrammetry.sh)
EOF
}

# Ejecuta (o muestra en DRY_RUN) un script Python con objetos y --overwrite según FORCE.
run_py() { # interprete script [extra...] -- objetos...
    local interp="$1" script="$2"; shift 2
    local extra=()
    while [[ $# -gt 0 && "$1" != -- ]]; do extra+=("$1"); shift; done
    [[ "${1:-}" == -- ]] && shift
    local cmd=("$interp" "$script" "$@" "${extra[@]}")
    [[ "$FORCE" == 1 ]] && cmd+=(--overwrite)
    log "\$ ${cmd[*]}"
    [[ "$DRY_RUN" == 1 ]] && return 0
    ( cd "$ROOT" && "${cmd[@]}" )
}

isaac() { # script objetos...
    [[ "$DRY_RUN" == 1 || -x "$ISAAC_PYTHON" ]] || die "No se encontró Isaac Sim en ISAAC_PYTHON=$ISAAC_PYTHON"
    local script="$1"; shift
    run_py "$ISAAC_PYTHON" "$script" -- "$@"
}

methods_args() { [[ -n "$METHODS" ]] && printf '%s\n' --methods "$METHODS"; return 0; }

photogrammetry() {
    local path="$PATH"
    if ! command -v colmap >/dev/null && [[ -x "$COLMAP_BIN_DIR/colmap" ]]; then
        path="$COLMAP_BIN_DIR:$PATH"
    fi
    [[ "$FORCE" == 1 ]] && export FORCE_PHOTOGRAMMETRY="${FORCE_PHOTOGRAMMETRY:-1}"
    PATH="$path" bash "$RECON/recon_photogrammetry.sh" "$@"
}

stage() { # etapa objetos...
    local name="$1"; shift
    local extra=()
    case "$name" in
        prepare)        isaac "$SRC/capture/prepare_scenes.py" "$@" ;;
        rgbd)           isaac "$SRC/capture/generate_rgbd_three_cameras.py" "$@" ;;
        orbit)          isaac "$SRC/capture/generate_orbit_room.py" "$@" ;;
        calibration)    isaac "$RECON/cameras/generate_calibration_views.py" "$@" ;;
        gt)             isaac "$SRC/ground_truth/usdz_to_plt.py" "$@" ;;
        process)        bash "$RECON/recon_process.sh" "$@" ;;
        nerf)           bash "$RECON/recon_nerf.sh" "$@" ;;
        splat)          bash "$RECON/recon_splat.sh" "$@" ;;
        recon)          bash "$RECON/recon_all.sh" "$@" ;;
        photogrammetry) photogrammetry "$@" ;;
        rgbd-fusion)    run_py "$PYTHON" "$RECON/rgbd_fusion.py" -- "$@" ;;
        clean)          mapfile -t extra < <(methods_args)
                        run_py "$PYTHON" "$SRC/evaluation/clean_reconstructions.py" "${extra[@]}" -- "$@" ;;
        evaluate)       mapfile -t extra < <(methods_args)
                        run_py "$PYTHON" "$SRC/evaluation/run_evaluation.py" "${extra[@]}" -- "$@" ;;
        *) die "Etapa desconocida: $name (./run.sh help)" ;;
    esac
}

# Ejecuta varias etapas seguidas; un fallo se reporta pero no detiene las siguientes.
sequence() { # "etapa1 etapa2 ..." objetos...
    local stages="$1"; shift
    local failed=() s
    for s in $stages; do
        log "################ $s ################"
        stage "$s" "$@" || { failed+=("$s"); log "La etapa $s terminó con fallos; se continúa."; }
    done
    log "Etapas con fallos: ${failed[*]:-ninguna}"
    [[ ${#failed[@]} -eq 0 ]]
}

main() {
    [[ $# -ge 1 ]] || { usage; exit 1; }
    local cmd="$1"; shift
    case "$cmd" in
        help|-h|--help) usage ;;
        status)  ( cd "$ROOT" && "$PYTHON" scripts/status.py "$@" ) ;;
        capture) sequence "prepare rgbd orbit calibration" "$@" ;;
        post)    sequence "gt clean evaluate" "$@" ;;
        all)     sequence "prepare rgbd orbit calibration recon photogrammetry rgbd-fusion gt clean evaluate" "$@" ;;
        *)       stage "$cmd" "$@" ;;
    esac
}

main "$@"
