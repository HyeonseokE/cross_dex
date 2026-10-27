#!/bin/bash
# Start the Manus bridge: owns the glove dongle (Manus SDK, Integrated mode) and streams the gloves over
# TCP (default 127.0.0.1:47000). Run it where the dongle is plugged in; only one program can hold the dongle.
#
#   vr_teleop/tools/run_manus_bridge.sh            # foreground
#   vr_teleop/tools/run_manus_bridge.sh --fake     # synthetic hand, no gloves (pipeline check)
#   vr_teleop/tools/run_manus_bridge.sh --bind 0.0.0.0 --port 47000
#
# MANUS_SDK_PY: the Manus SDK Python folder (default /workspace/manus_sdk/Python).
# Check what arrives: python vr_teleop/devices/manus/bridge/manus_bridge_client.py
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MANUS_SDK_PY=${MANUS_SDK_PY:-/workspace/manus_sdk/Python}
exec python -u "${HERE}/../devices/manus/bridge/manus_bridge_server.py" "$@"
