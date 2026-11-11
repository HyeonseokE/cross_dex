"""Cross-embodiment paired chunk dataset spanning the grasp taxonomy (Feix GRASP, 33 types).

Human hand motion (DexCanvas, HOGraspNet) and 5-finger robot grasps (Dexonomy) are put into one representation,
labelled with Feix grasp types, retargeted to every 5-finger hand with the DexVerse teleop retargeter itself,
resampled to the policy rate and cut into H-step chunks that pair across hands (same source motion, same time window).
Construction steps and design: RSS2027 bundle ``ablation/dexverse-chunk-latent/DATASET_PIPELINE.md``.
"""
