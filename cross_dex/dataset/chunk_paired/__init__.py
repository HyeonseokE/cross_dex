"""Cross-embodiment paired chunk dataset spanning the grasp taxonomy (Feix GRASP, 33 types).

Human hand motion (DexCanvas, HOGraspNet) and 5-finger robot grasps (Dexonomy) are put into one representation,
labelled with Feix grasp types, retargeted to every 5-finger hand with the DexVerse teleop retargeter itself,
resampled to the policy rate and cut into H-step chunks that pair across hands (same source motion, same time window).
Construction steps and design: RSS2027 bundle ``ablation/dexverse-chunk-latent/DATASET_PIPELINE.md``.
"""

import os as _os

# Generated data (per source: dexcanvas/, hograspnet/, dexonomy/, transition/). On servers this is a link to
# /workspace/datasets/cross_dex/chunk_paired_dataset (data stays out of the synced code copy, git-ignored).
DATA_ROOT = _os.environ.get("CHUNK_PAIRED_DATA", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "chunk_paired_dataset"))
