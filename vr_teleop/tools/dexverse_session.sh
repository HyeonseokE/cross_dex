#!/bin/bash
# DexVerse teleop / recording with the Manus gloves + glove-mounted Quest controller, standard settings.
#
#   vr_teleop/tools/dexverse_session.sh teleop --task Dexverse-GraspCup-v1 [--robot_type floating_shadow_right]
#   vr_teleop/tools/dexverse_session.sh record --task Dexverse-GraspCup-v1 --num_demos 50
#
# Before: the CloudXR runtime is up (DexVerse scripts/teleop_tools/start_cloudxr_runtime.sh) and the glove
# calibration exists ($VR_TELEOP_CALIB, see vr_teleop/tools/calibration). This script starts the Manus bridge
# if it is not running, then runs DexVerse teleop_agent.py / record_demos.py with --teleop_device manus.
# Defaults (override in the environment): VR_TELEOP_WRIST=controller, VR_TELEOP_CALIB, robot floating_shadow_right;
# record adds --record_state. Other arguments are passed through.
set -eo pipefail
MODE=${1:?usage: dexverse_session.sh <teleop|record> --task <task> [args...]}
shift
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEXVERSE_ROOT=${DEXVERSE_ROOT:-$(cd "${HERE}/../../.." && pwd)}
LOG_DIR=${VR_TELEOP_LOG_DIR:-/workspace/logs/teleop/$(date +%Y%m%d)}
mkdir -p "${LOG_DIR}"

export VR_TELEOP_WRIST=${VR_TELEOP_WRIST:-controller}
: "${VR_TELEOP_CALIB:?set VR_TELEOP_CALIB to the calibration json (see vr_teleop/tools/calibration)}"
export VR_TELEOP_CALIB
STAMP=$(date +%H%M%S)
export VR_TELEOP_TRACE=${VR_TELEOP_TRACE-${LOG_DIR}/controller_${STAMP}.csv}
export PYTHONUNBUFFERED=1  # the log goes through tee: print immediately

if ! (exec 3<>/dev/tcp/127.0.0.1/${VR_TELEOP_BRIDGE_PORT:-47000}) 2>/dev/null; then
    echo "[session] starting the Manus bridge (log ${LOG_DIR}/manus_bridge.log)"
    setsid nohup "${HERE}/run_manus_bridge.sh" >> "${LOG_DIR}/manus_bridge.log" 2>&1 &
    ln -sfn "${LOG_DIR}/manus_bridge.log" "$(dirname "${LOG_DIR}")/manus_bridge.log"  # stable path: the running bridge's log
    sleep 3
fi

ARGS=("$@")
has_arg() { local a; for a in "${ARGS[@]}"; do [[ "$a" == "$1" || "$a" == "$1="* ]] && return 0; done; return 1; }
has_arg --robot_type || ARGS+=(--robot_type floating_shadow_right)
case "${MODE}" in
    teleop) SCRIPT=teleop_agent ;;
    record) SCRIPT=record_demos; has_arg --record_state || ARGS+=(--record_state) ;;
    *) echo "unknown mode '${MODE}' (teleop | record)" >&2; exit 1 ;;
esac
echo "[session] ${SCRIPT} wrist=${VR_TELEOP_WRIST} calib=${VR_TELEOP_CALIB} log=${LOG_DIR}/${SCRIPT}_${STAMP}.log trace=${VR_TELEOP_TRACE}"
cd "${DEXVERSE_ROOT}"
"${DEXVERSE_ROOT}/scripts/teleop_tools/run_teleop.sh" "${SCRIPT}" --teleop_device manus "${ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/${SCRIPT}_${STAMP}.log"
