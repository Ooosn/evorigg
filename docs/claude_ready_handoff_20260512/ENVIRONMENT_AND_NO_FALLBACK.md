# Environment And No-Fallback Rules

## Required environment

Use this environment for official runs:

- `conda activate mygs`
- Python: `D:\Users\namew\miniconda3\envs\mygs\python.exe`

Mandatory preflight:

```powershell
conda activate mygs
D:\Users\namew\miniconda3\envs\mygs\python.exe scripts\check_evorig_environment.py
```

The preflight must return:

- `status == ok`
- `cuda.available == true`
- repo imports succeed

## Components that must be installed

These must exist and import successfully:

- `numpy`
- `PyYAML`
- `torch`
- `tqdm`
- `trimesh`
- `scipy`
- `open3d`
- `plotly`

These capabilities must exist:

- `torch.cuda.is_available() == True`
- `open3d.t.geometry.RaycastingScene`

These external tools must exist when preparing/importing UniRig samples:

- Blender executable
  - preferred path: `D:\Program Files\Blender Foundation\Blender 5.0\blender.exe`
  - otherwise pass explicit `--blender-path`

## Hard no-fallback rules

The following fallbacks are forbidden for official experiments:

- No CPU fallback for training
  - do not use `--no-cuda-required` except for diagnostics
- No missing-module fallback
  - if preflight fails, fix the environment first
- No viewer semantic fallback
  - viewer must read `phase1_config.json` or checkpoint-embedded config
  - if config is missing, fail; do not guess legacy defaults
- No skeleton-only UniRig import fallback
  - official UniRig dynamic samples must import from `rigged.glb`
  - `skeleton.fbx` is used only to overlay Blender `use_connect`
  - do not use `raw_data.npz` as a skeleton source for official runs
- No fallback to old sample variants
  - for current `double_knife`, do not use `v1`, `v2`, or `v3`
- No mixing UniRig and ActionMesh import paths
  - `import_unirig_dynamic_glb_sample.py` is for `E:\evorig_unirig`
  - `import_real_glb_sample.py` is the separate ActionMesh/real-GLB path
- No silent fallback to legacy code/config paths
  - use only the active frozen configs listed in this package

## Connected-bone rule

`connected_to_parent=false` means:

- keep hierarchy for FK
- draw as dashed hierarchy link
- do not treat it as a physical support bone
- do not initialize Gaussians on that link
- do not include that link in PCJS / cross-section / posed-bone-inside support logic

If a sample does not preserve this from `skeleton.fbx`, the sample is not
authoritative and must be rebuilt.
