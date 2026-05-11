# Import Paths

There are two supported mesh-import families. They are intentionally separate.
Do not mix their commands, assumptions, or outputs.

## A. Current default: UniRig dynamic mesh samples

Use this path for the current comparison experiments under:

- `E:\evorig_unirig\<mesh_name>\dynamic_mesh.glb`

Required input/output flow:

1. Bake the selected motion window from `dynamic_mesh.glb`.
2. Write the selected rest frame to `mesh.glb`.
3. Run UniRig on `mesh.glb`.
4. Require these files in the same asset folder:
   - `dynamic_mesh.glb`
   - `rigged.glb`
   - `skeleton.fbx`
   - `skin.fbx`
5. Build the EvoRig sample with:
   - `scripts/import_unirig_dynamic_glb_sample.py`

Current semantics:

- `rigged.glb` is the aligned source for rest mesh, joint positions, and skin
  weights.
- `skeleton.fbx` is used only to overlay Blender `bone.use_connect` into
  `connected_to_parent`.
- Non-connected parent links are hierarchy links only: draw dashed, do not
  initialize Gaussians, and do not use them as physical support bones.
- The old skeleton-only UniRig importer is deleted and must not be restored as
  an official path.

## B. Separate legacy-compatible path: ActionMesh / real GLB samples

Use this path only for older assets that already provide explicit animated and
rigged GLB inputs, such as the ActionMesh-style layout:

- `animated_mesh.glb`
- `rigged.glb` or an explicitly passed rigged asset

Entry point:

- `scripts/import_real_glb_sample.py`

Current semantics:

- This path is separate from the UniRig dynamic protocol.
- It does not run UniRig.
- It does not use `skeleton.fbx` `use_connect` unless the importer is explicitly
  extended for that asset family.
- It is not the default path for `E:\evorig_unirig` experiments.

## Quick decision rule

- If the sample starts from `E:\evorig_unirig\<mesh>\dynamic_mesh.glb`, use the
  UniRig dynamic path.
- If the sample already has a rigged GLB and animated GLB from the old/action
  mesh pipeline, use `import_real_glb_sample.py`.
- If alignment/normalization gates fail, stop. Do not train.
