#!/bin/bash
# Put the multi-hand DP inputs on a server that has none (2026-11-20: vast-isaac2), from the private HF relay
# Dexverse-CSI/relay-xhdp (uploaded from p1): merged training sets (data/meta per arm, one video copy per set, symlinked),
# GraspKettle teleop pickles + teleop set meta (action names), wrist calibration, space checkpoints, DexVerse assets.
#   echo <hf token> | bash setup_server.sh          (token from stdin, not stored)
set -euo pipefail
read T; export HF_TOKEN=$T HF_HOME=/workspace/.cache/huggingface
R=/workspace/.tmp/relay_xhdp; D=/workspace/datasets/dexverse
/workspace/envs/dexverse/dexlatent/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Dexverse-CSI/relay-xhdp', repo_type='dataset', local_dir='$R', allow_patterns=['inputs.tar','assets_dexverse.tar'], max_workers=4)"
tar -C $R -xf $R/inputs.tar && rm -f $R/inputs.tar   # one file: the per-episode meta files hit the HF request limit (1000 / 5 min)
mkdir -p $D/crosshand_dp/_videos $D/teleop_pkl_bs1/functional/Dexverse-GraspKettle-v0 $D/teleop_lerobot_goal/Dexverse-GraspKettle-v0
for s in a6 b_noshadow; do rm -rf $D/crosshand_dp/_videos/$s; cp -r $R/videos/$s $D/crosshand_dp/_videos/$s; done
for a in xlvla ea eh e2c joint; do for s in a6 b_noshadow; do
  t=$D/crosshand_dp/$a/$s; rm -rf $t; mkdir -p $t; cp -r $R/sets/$a/$s/data $R/sets/$a/$s/meta $t/; ln -s ../../_videos/$s $t/videos
done; done
cp $R/pkl/*.pkl $D/teleop_pkl_bs1/functional/Dexverse-GraspKettle-v0/
cp -r $R/teleop_meta/* $D/teleop_lerobot_goal/Dexverse-GraspKettle-v0/
mkdir -p /workspace/checkpoints/dexverse/xlvla/hf6/latent /workspace/checkpoints/dexverse/crosshand_dp
cp -r $R/wrist /workspace/checkpoints/dexverse/xlvla/hf6/latent/; cp -r $R/spaces /workspace/checkpoints/dexverse/crosshand_dp/
mkdir -p /workspace/assets; tar -C /workspace/assets -xf $R/assets_dexverse.tar
rm -f $R/assets_dexverse.tar
echo "[setup] inputs done"
