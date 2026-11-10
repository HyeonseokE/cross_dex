#!/bin/bash
# Bench2Dex teleop / demo collection with the Manus gloves + glove-mounted Quest controllers (same gear and
# calibration as the DexVerse sessions).
#
#   cross_dex/cross_dex/bench2dex/adapters/b2d_session.sh teleop --task scenes/06_fruit_bowl_loading.yaml
#   cross_dex/cross_dex/bench2dex/adapters/b2d_session.sh record --task scenes/06_fruit_bowl_loading.yaml
#
# Before: the CloudXR runtime is up (port 48322) and the glove calibration exists ($VR_TELEOP_CALIB). This script
# starts the Manus bridge if it is not running, then runs bench2dex main.py --teleop --teleop-device manus-quest
# (record = --collect). Headset buttons: START = home + record, STOP = home + save, RESET = discard + home.
# Settings: B2D_ROOT (bench2dex checkout, default /workspace/bench2dex), B2D_ENV_SH (activation script, default
# /workspace/bench2dex_env.sh if present), VR_TELEOP_CALIB, VR_TELEOP_BRIDGE_PORT (47000), VR_TELEOP_LOG_DIR.
# Other arguments are passed through to main.py (e.g. --xr-pos-scale 1.0, --gui).
set -eo pipefail
MODE=${1:?usage: b2d_session.sh <teleop|record> --task <scene yaml> [main.py args...]}
shift
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
B2D_ROOT=${B2D_ROOT:-/workspace/bench2dex}
B2D_ENV_SH=${B2D_ENV_SH:-/workspace/bench2dex_env.sh}
[ -f "${B2D_ENV_SH}" ] && source "${B2D_ENV_SH}"
LOG_DIR=${VR_TELEOP_LOG_DIR:-/workspace/logs/b2d_teleop/$(date +%Y%m%d)}
mkdir -p "${LOG_DIR}"
: "${VR_TELEOP_CALIB:?set VR_TELEOP_CALIB to the calibration json (see cross_dex/teleop/tools/calibration)}"
export VR_TELEOP_CALIB
STAMP=$(date +%H%M%S)
export VR_TELEOP_TRACE=${VR_TELEOP_TRACE-${LOG_DIR}/controller_${STAMP}.csv}
export PYTHONUNBUFFERED=1

CXR_ENV=${CXR_ENV:-${CXR_HOME:-$HOME/.cloudxr}/run/cloudxr.env}
if ! (exec 3<>"/dev/tcp/127.0.0.1/${CXR_WSS_PORT:-48322}") 2>/dev/null; then
    echo "[b2d-session] CloudXR runtime is not running (port 48322): start start_cloudxr_runtime.sh first." >&2
    exit 1
fi
if ! (exec 3<>/dev/tcp/127.0.0.1/${VR_TELEOP_BRIDGE_PORT:-47000}) 2>/dev/null; then
    echo "[b2d-session] starting the Manus bridge (log ${LOG_DIR}/manus_bridge.log)"
    setsid nohup "${HERE}/../../teleop/tools/run_manus_bridge.sh" >> "${LOG_DIR}/manus_bridge.log" 2>&1 &
    ln -sfn "${LOG_DIR}/manus_bridge.log" "$(dirname "${LOG_DIR}")/manus_bridge.log"
    sleep 3
fi

ARGS=("$@")
has_arg() { local a; for a in "${ARGS[@]}"; do [[ "$a" == "$1" || "$a" == "$1="* ]] && return 0; done; return 1; }
headless=1
if has_arg --gui; then
    headless=0; filtered=(); for a in "${ARGS[@]}"; do [[ "$a" != "--gui" ]] && filtered+=("$a"); done; ARGS=("${filtered[@]}")
fi
case "${MODE}" in
    teleop) ;;
    record) has_arg --collect || ARGS+=(--collect) ;;
    *) echo "unknown mode '${MODE}' (teleop | record)" >&2; exit 1 ;;
esac
has_arg --teleop || ARGS+=(--teleop)
has_arg --teleop-device || ARGS+=(--teleop-device manus-quest)
has_arg --xr-stream-log || ARGS+=(--xr-stream-log 100)
if [[ ${headless} == 1 ]] && ! has_arg --headless; then ARGS+=(--headless); fi
if [ -f "${CXR_ENV}" ]; then set -a; source "${CXR_ENV}"; set +a; fi
unset PYTHONPATH
cd "${B2D_ROOT}"
echo "[b2d-session] ${MODE} calib=${VR_TELEOP_CALIB} log=${LOG_DIR}/main_${STAMP}.log trace=${VR_TELEOP_TRACE}"
echo "[b2d-session] python main.py ${ARGS[*]}"
python main.py "${ARGS[@]}" 2>&1 | tee -a "${LOG_DIR}/main_${STAMP}.log"
