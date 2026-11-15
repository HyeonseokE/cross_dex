# dexjoco_urdf

Kinematics-only URDFs (no meshes) of the seven DexJoCo-X hands, right and left, written from the compiled DexJoCo-X MuJoCo
models by `../dexjoco_mjcf2urdf.py` (source: https://github.com/dexjoco-x/DexJoCo-X-public, `benchmark7_v3_bimanual_photograph_<hand>.xml`).
`<hand>_<side>.json` = actuator-order joint names and fingertip links; `dataset_joint_names.json` = joint names of the pair files.

The hand models are third-party assets of DexJoCo-X and of the hand makers; they keep their original licences
(DexJoCo-X `THIRD_PARTY_NOTICES.md`, `LICENSE_STATUS.md`). Regenerate with the script instead of editing these files.
