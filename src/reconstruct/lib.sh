#!/usr/bin/env bash
# Funciones comunes. Se usa con "source"; no se ejecuta directamente.

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$LIB_DIR/../.." && pwd)}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
NERFSTUDIO_DIR="${NERFSTUDIO_DIR:-$PROJECT_ROOT/nerfstudio}"
REPORT_DIR="${REPORT_DIR:-$PROJECT_ROOT/reports}"

# ================= CONFIGURACION (se puede sobrescribir con variables de entorno) =================
NUM_FRAMES="${NUM_FRAMES:-300}"             # Imagenes extraidas del video
MIN_REGISTERED_FRACTION="${MIN_REGISTERED_FRACTION:-0.90}"  # Minimo de camaras que COLMAP debe registrar
NERF_ITERS="${NERF_ITERS:-2000}"            # Valor de prueba; fijar uno comun para el experimento final
SPLAT_ITERS="${SPLAT_ITERS:-4000}"          # Valor de prueba; fijar uno comun para el experimento final
NERF_EXPORT_POINTS="${NERF_EXPORT_POINTS:-500000}"
NERF_RAYS_PER_BATCH="${NERF_RAYS_PER_BATCH:-8192}"
GCC_BIN="${GCC_BIN:-/usr/bin/gcc-11}"
GXX_BIN="${GXX_BIN:-/usr/bin/g++-11}"
MAX_JOBS_CUDA="${MAX_JOBS_CUDA:-4}"
FORCE="${FORCE:-0}"                         # 1 = rehacer etapas ya terminadas
SKIP_EXPORT="${SKIP_EXPORT:-0}"             # 1 = solo entrenar
DRY_RUN="${DRY_RUN:-0}"                     # 1 = solo mostrar que se haria
# ==================================================================================================

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

check_environment() {
  [[ -d "$DATA_DIR" ]] || die "No existe DATA_DIR: $DATA_DIR"
  [[ -d "$NERFSTUDIO_DIR" ]] || die "No existe NERFSTUDIO_DIR: $NERFSTUDIO_DIR"
  command -v pixi >/dev/null || die "pixi no esta en PATH"
  command -v python3 >/dev/null || die "python3 no esta en PATH"
  command -v flock >/dev/null || die "flock no esta disponible"
  if [[ "$DRY_RUN" != 1 ]]; then
    mkdir -p "$REPORT_DIR"
    exec 9>"$PROJECT_ROOT/.recon.lock"
    flock -n 9 || die "Ya hay otra reconstruccion usando este proyecto/GPU"
  fi
}

# Ejecuta un comando dentro del proyecto Pixi de Nerfstudio.
ns() { ( cd "$NERFSTUDIO_DIR" && "$@" ); }

# Ejecuta y guarda salida en el log indicado.
run_logged() {
  local logfile="$1"; shift
  mkdir -p "$(dirname "$logfile")"
  { printf '$'; printf ' %q' "$@"; printf '\n'; } >> "$logfile"
  "$@" 2>&1 | tee -a "$logfile"
}

discover_objects() {
  if [[ $# -gt 0 ]]; then
    for obj in "$@"; do
      obj="${obj%.usdz}"
      [[ -f "$DATA_DIR/$obj/raw/orbit.mp4" ]] || { log "OMITIDO $obj: falta $DATA_DIR/$obj/raw/orbit.mp4" >&2; continue; }
      echo "$obj"
    done
  else
    for dir in "$DATA_DIR"/*/; do
      [[ -f "${dir}raw/orbit.mp4" ]] && basename "$dir"
    done
  fi
}

record() {  # objeto etapa estado segundos detalle
  local obj="$1" stage="$2" status="$3" secs="$4" detail="${5:-}"
  local rec="$DATA_DIR/$obj/reconstruction"
  mkdir -p "$rec" "$REPORT_DIR"
  for f in "$rec/stages.csv" "$REPORT_DIR/reconstruction_summary.csv"; do
    [[ -f "$f" ]] || echo "timestamp,object,stage,status,seconds,detail" > "$f"
    printf '%s,%s,%s,%s,%s,"%s"\n' "$(date -Is)" "$obj" "$stage" "$status" "$secs" "$detail" >> "$f"
  done
}

# timed_stage objeto etapa funcion [args...]
# La funcion puede poner STAGE_STATUS=skipped y STAGE_DETAIL.
timed_stage() {
  local obj="$1" stage="$2"; shift 2
  local t0=$SECONDS
  STAGE_STATUS=ok; STAGE_DETAIL=""
  log "=== $obj :: $stage ==="
  if [[ "$DRY_RUN" == 1 ]]; then log "(dry-run) $*"; return 0; fi
  if "$@"; then
    record "$obj" "$stage" "$STAGE_STATUS" "$((SECONDS - t0))" "$STAGE_DETAIL"
    return 0
  fi
  record "$obj" "$stage" failed "$((SECONDS - t0))" "$STAGE_DETAIL"
  log "FALLO: $obj :: $stage"
  return 1
}

# ---------------------------------------------------------------- procesamiento (COLMAP)
validate_processed() {  # directorio
  python3 - "$1" "$MIN_REGISTERED_FRACTION" <<'PY'
import json, sys
from pathlib import Path
p, minimo = Path(sys.argv[1]), float(sys.argv[2])
try:
    t = json.loads((p / "transforms.json").read_text())
except Exception as e:
    print(f"transforms.json no valido: {e}"); sys.exit(1)
registradas = len(t.get("frames", []))
imagenes = len(list((p / "images").glob("*"))) if (p / "images").is_dir() else 0
print(f"camaras registradas={registradas} imagenes={imagenes}")
if imagenes == 0 or registradas / imagenes < minimo:
    print(f"Registro insuficiente (minimo {minimo:.0%}); no se entrena."); sys.exit(1)
fx, fy = t.get("fl_x"), t.get("fl_y")
if fx and fy and not (0.5 <= fx / fy <= 2.0):
    print(f"Intrinsecos sospechosos: fl_x={fx}, fl_y={fy}"); sys.exit(1)
PY
}

stage_process() {  # objeto
  local obj="$1"
  local base="$DATA_DIR/$obj" video="$DATA_DIR/$obj/raw/orbit.mp4"
  local rec="$base/reconstruction" proc="$base/reconstruction/processed"
  local logfile="$rec/logs/process.log"
  if [[ -f "$rec/.process.done" && "$FORCE" != 1 ]]; then
    STAGE_STATUS=skipped; STAGE_DETAIL="ya procesado"; return 0
  fi
  # Lo procesado es derivado del video: se puede regenerar. Se invalidan entrenamientos asociados.
  rm -rf "$proc" "$rec"/.process.done "$rec"/.train_*.done "$rec"/.export_*.done
  mkdir -p "$rec/logs"
  export QT_QPA_PLATFORM=offscreen
  run_logged "$logfile" ns pixi run ns-process-data video \
      --data "$video" --output-dir "$proc" \
      --num-frames-target "$NUM_FRAMES" || { STAGE_DETAIL="ns-process-data fallo"; return 1; }
  STAGE_DETAIL="$(validate_processed "$proc" | tee -a "$logfile" | tr '\n' ' ')"
  validate_processed "$proc" >/dev/null || return 1
  date -Is > "$rec/.process.done"
}

# ---------------------------------------------------------------- entrenamiento
latest_config() {  # directorio_metodo
  find "$1" -name config.yml -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-
}

stage_train() {  # objeto metodo(nerfacto|splatfacto) iteraciones
  local obj="$1" method="$2" iters="$3"
  local rec="$DATA_DIR/$obj/reconstruction"
  local marker="$rec/.train_$method.done" logfile="$rec/logs/train_$method.log"
  if [[ -f "$marker" && "$FORCE" != 1 ]]; then
    STAGE_STATUS=skipped; STAGE_DETAIL="ya entrenado"; return 0
  fi
  [[ -f "$rec/.process.done" ]] || { STAGE_DETAIL="falta procesamiento"; return 1; }
  rm -f "$marker"
  local cmd=(pixi run ns-train "$method"
      --data "$rec/processed" --output-dir "$rec/runs" --experiment-name "$obj"
      --max-num-iterations "$iters" --viewer.quit-on-train-completion True)
  if [[ "$method" == splatfacto ]]; then
    CC="$GCC_BIN" CXX="$GXX_BIN" CUDAHOSTCXX="$GXX_BIN" MAX_JOBS="$MAX_JOBS_CUDA" \
      run_logged "$logfile" ns "${cmd[@]}" || { STAGE_DETAIL="ns-train fallo"; return 1; }
  else
    run_logged "$logfile" ns "${cmd[@]}" || { STAGE_DETAIL="ns-train fallo"; return 1; }
  fi
  local config; config="$(latest_config "$rec/runs/$obj/$method")"
  [[ -f "$config" ]] || { STAGE_DETAIL="no se encontro config.yml"; return 1; }
  echo "$config" > "$marker"
  STAGE_DETAIL="iters=$iters config=$config"
}

# ---------------------------------------------------------------- exportacion
stage_export_nerf() {  # objeto
  local obj="$1" rec="$DATA_DIR/$obj/reconstruction"
  local marker="$rec/.export_nerfacto.done" logfile="$rec/logs/export_nerfacto.log"
  if [[ -f "$marker" && "$FORCE" != 1 ]]; then STAGE_STATUS=skipped; return 0; fi
  [[ -f "$rec/.train_nerfacto.done" ]] || { STAGE_DETAIL="falta entrenamiento"; return 1; }
  local config out="$rec/exports/nerfacto"
  config="$(cat "$rec/.train_nerfacto.done")"
  rm -rf "$out"; rm -f "$marker"
  run_logged "$logfile" ns pixi run ns-export pointcloud \
      --load-config "$config" --output-dir "$out" \
      --num-points "$NERF_EXPORT_POINTS" --normal-method open3d \
      --remove-outliers False --num-rays-per-batch "$NERF_RAYS_PER_BATCH" \
      || { STAGE_DETAIL="ns-export fallo"; return 1; }
  [[ -s "$out/point_cloud.ply" ]] || { STAGE_DETAIL="falta point_cloud.ply"; return 1; }
  date -Is > "$marker"
}

stage_export_splat() {  # objeto
  local obj="$1" rec="$DATA_DIR/$obj/reconstruction"
  local marker="$rec/.export_splatfacto.done" logfile="$rec/logs/export_splatfacto.log"
  if [[ -f "$marker" && "$FORCE" != 1 ]]; then STAGE_STATUS=skipped; return 0; fi
  [[ -f "$rec/.train_splatfacto.done" ]] || { STAGE_DETAIL="falta entrenamiento"; return 1; }
  local config out="$rec/exports/splatfacto"
  config="$(cat "$rec/.train_splatfacto.done")"
  rm -rf "$out"; rm -f "$marker"
  CC="$GCC_BIN" CXX="$GXX_BIN" CUDAHOSTCXX="$GXX_BIN" MAX_JOBS="$MAX_JOBS_CUDA" \
    run_logged "$logfile" ns pixi run ns-export gaussian-splat \
      --load-config "$config" --output-dir "$out" \
      || { STAGE_DETAIL="ns-export fallo"; return 1; }
  local plys=("$out"/*.ply)
  [[ ${#plys[@]} -eq 1 && -s "${plys[0]}" ]] || { STAGE_DETAIL="se esperaba un unico PLY"; return 1; }
  # Centros de las gaussianas (para inspeccion; NO es una superficie)
  ns pixi run python "$LIB_DIR/splat_centers.py" "${plys[0]}" "$out/gaussian_centers_NOT_SURFACE.ply" \
      >> "$logfile" 2>&1 || log "AVISO: no se pudo generar el PLY de centros (revisa $logfile)"
  date -Is > "$marker"
}

# ---------------------------------------------------------------- flujo principal
run_pipeline() {  # "nerf splat" objetos...
  local methods="$1"; shift
  check_environment
  mapfile -t objects < <(discover_objects "$@")
  [[ ${#objects[@]} -gt 0 ]] || die "No hay objetos con raw/orbit.mp4 en $DATA_DIR"
  log "Objetos (${#objects[@]}): ${objects[*]}"
  local failed=()
  for obj in "${objects[@]}"; do
    {
      timed_stage "$obj" process stage_process "$obj" || exit 1
      if [[ " $methods " == *" nerf "* ]]; then
        timed_stage "$obj" train_nerfacto stage_train "$obj" nerfacto "$NERF_ITERS" || exit 1
        [[ "$SKIP_EXPORT" == 1 ]] || timed_stage "$obj" export_nerfacto stage_export_nerf "$obj" || exit 1
      fi
      if [[ " $methods " == *" splat "* ]]; then
        timed_stage "$obj" train_splatfacto stage_train "$obj" splatfacto "$SPLAT_ITERS" || exit 1
        [[ "$SKIP_EXPORT" == 1 ]] || timed_stage "$obj" export_splatfacto stage_export_splat "$obj" || exit 1
      fi
    } || failed+=("$obj")
  done
  log "Terminado. Fallos: ${#failed[@]} ${failed[*]:-}"
  [[ ${#failed[@]} -eq 0 ]]
}
