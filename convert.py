# convert.py
from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import os

import numpy as np
import trimesh

# =========================
# Config
# =========================
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"


# =========================
# Blender: GLB -> OBJ
# =========================
def glb_to_obj_blender(input_glb: str, output_obj: str) -> None:
    """
    Convert GLB/GLTF -> OBJ using Blender headless.
    - Robust on Windows (utf-8 stdout/stderr)
    - Exports geometry only (no materials)
    """
    input_glb_p = Path(input_glb).resolve()
    output_obj_p = Path(output_obj).resolve()
    output_obj_p.parent.mkdir(parents=True, exist_ok=True)

    if not input_glb_p.exists():
        raise FileNotFoundError(f"Input GLB not found: {input_glb_p}")

    if not Path(BLENDER_EXE).exists():
        raise FileNotFoundError(f"Blender exe not found: {BLENDER_EXE}")

    # Inline blender script
    blender_script = f"""
import bpy
import os

inp = r"{str(input_glb_p)}"
out_obj = r"{str(output_obj_p)}"

bpy.ops.wm.read_factory_settings(use_empty=True)

# Import glTF/GLB
bpy.ops.import_scene.gltf(filepath=inp)

# Select meshes
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not meshes:
    raise RuntimeError("No mesh imported from GLB")

# Join meshes
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()

# Apply transforms
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Export OBJ
# Blender 5.x: wm.obj_export is preferred
if hasattr(bpy.ops.wm, "obj_export"):
    bpy.ops.wm.obj_export(
        filepath=out_obj,
        export_selected_objects=True,
        export_materials=False,
    )
else:
    bpy.ops.export_scene.obj(
        filepath=out_obj,
        use_selection=True,
        use_materials=False,
    )

print("[OK] OBJ written:", out_obj)
"""

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(blender_script)
        script_path = f.name

    cmd = [BLENDER_EXE, "--background", "--python", script_path]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    try:
        os.remove(script_path)
    except Exception:
        pass

    if r.stdout:
        print("[BLENDER] STDOUT:\n", r.stdout)
    if r.stderr:
        print("[BLENDER] STDERR:\n", r.stderr)

    if r.returncode != 0:
        raise RuntimeError(f"Blender GLB->OBJ failed (code={r.returncode})")

    if not output_obj_p.exists():
        raise RuntimeError(f"OBJ not created: {output_obj_p}")


# =========================
# Print prep helpers
# =========================
def scale_to_height_mm(mesh: trimesh.Trimesh, target_mm: float) -> None:
    """
    Scale mesh so that Z height becomes target_mm.
    Assumes mesh is already in mm scale.
    """
    zmin = float(mesh.bounds[0][2])
    zmax = float(mesh.bounds[1][2])
    height = zmax - zmin
    if height <= 1e-9:
        raise RuntimeError("Invalid mesh height")
    s = float(target_mm) / float(height)
    mesh.apply_scale(s)


def drop_to_ground(mesh: trimesh.Trimesh) -> None:
    """Translate mesh so that min Z becomes 0."""
    zmin = float(mesh.bounds[0][2])
    mesh.apply_translation([0.0, 0.0, -zmin])


def _overhang_area(mesh: trimesh.Trimesh, max_overhang_deg: float = 45.0) -> float:
    """
    Approx support-needed area proxy.
    +Z is "up". Faces whose normal Z component is small are more likely to need support.
    threshold = sin(max_overhang_deg) (45deg -> 0.707)
    """
    n = mesh.face_normals
    a = mesh.area_faces
    z = n[:, 2]
    thr = float(np.sin(np.deg2rad(max_overhang_deg)))
    mask = z < thr
    return float(a[mask].sum())


def _stability_penalty(mesh: trimesh.Trimesh) -> float:
    """
    Simple stability penalty proxy:
    - Use near-ground vertices to estimate base XY footprint
    - Penalize COM distance from footprint center
    """
    v = mesh.vertices
    zmin = float(v[:, 2].min())
    near = v[:, 2] < (zmin + 0.5)  # 0.5mm band
    if int(near.sum()) < 20:
        return 1e6

    pts = v[near][:, :2]
    minxy = pts.min(axis=0)
    maxxy = pts.max(axis=0)
    size = (maxxy - minxy)
    base_area_proxy = float(size[0] * size[1])
    if base_area_proxy <= 1e-9:
        return 1e6

    com = mesh.center_mass[:2]
    center = (minxy + maxxy) / 2.0
    dist = float(np.linalg.norm(com - center))
    return dist / (np.sqrt(base_area_proxy) + 1e-9)


def orient_for_print_support_min(
    mesh: trimesh.Trimesh,
    max_overhang_deg: float = 45.0,
    w_overhang: float = 1.0,
    w_stability: float = 0.35,
    max_poses: int = 12,
) -> dict:
    """
    Choose best stable pose that minimizes proxy support needs.
    score = w_overhang*overhang_area + w_stability*stability_penalty
    """
    poses = trimesh.poses.compute_stable_poses(mesh, sigma=0.0)
    if not poses:
        return {"chosen_index": None, "note": "no stable poses"}

    best = None
    best_meta = None

    for i, (T, prob) in enumerate(poses[:max_poses]):
        m = mesh.copy()
        m.apply_transform(T)
        drop_to_ground(m)

        oh = _overhang_area(m, max_overhang_deg=max_overhang_deg)
        st = _stability_penalty(m)
        score = w_overhang * oh + w_stability * st

        if best is None or score < best:
            best = score
            best_meta = (i, T, oh, st, score)

    idx, Tbest, oh, st, score = best_meta
    mesh.apply_transform(Tbest)
    return {
        "chosen_index": int(idx),
        "score": float(score),
        "overhang_area": float(oh),
        "stability_penalty": float(st),
        "max_overhang_deg": float(max_overhang_deg),
    }


def orient_for_print_simple(mesh: trimesh.Trimesh) -> dict:
    """Simple: pick the first stable pose."""
    poses = trimesh.poses.compute_stable_poses(mesh, sigma=0.0)
    if not poses:
        return {"chosen_index": None, "note": "no stable poses"}
    T, prob = poses[0]
    mesh.apply_transform(T)
    return {"chosen_index": 0, "note": "first stable pose"}


def add_base_shape(
    mesh: trimesh.Trimesh,
    base_mode: str = "circle",      # none|circle|oval|square|hex
    thickness_mm: float = 5.0,
    size_ratio: float = 0.75,
    margin_mm: float = 2.0,
    oval_xy_ratio: float = 1.35,
    hex_sides: int = 6,
) -> trimesh.Trimesh:
    """
    Add a base under the character.
    Assumes mesh already in mm and grounded (minZ=0).
    Implementation uses concatenate (fast, robust). Boolean union can be added later.
    """
    mode = (base_mode or "none").lower()
    if mode in ("none", "off", "false", "0"):
        return mesh

    bounds = mesh.bounds
    size_x = float(bounds[1][0] - bounds[0][0])
    size_y = float(bounds[1][1] - bounds[0][1])

    span = max(size_x, size_y) + 2.0 * float(margin_mm)
    base_size = float(span) * float(size_ratio)

    # Lift character on top of base
    mesh_up = mesh.copy()
    mesh_up.apply_translation([0.0, 0.0, float(thickness_mm)])

    if mode == "circle":
        radius = base_size / 2.0
        base = trimesh.creation.cylinder(radius=radius, height=float(thickness_mm), sections=96)
        base.apply_translation([0.0, 0.0, float(thickness_mm) / 2.0])

    elif mode == "oval":
        radius = base_size / 2.0
        base = trimesh.creation.cylinder(radius=radius, height=float(thickness_mm), sections=96)
        S = np.eye(4)
        S[0, 0] = float(oval_xy_ratio)
        base.apply_transform(S)
        base.apply_translation([0.0, 0.0, float(thickness_mm) / 2.0])

    elif mode == "square":
        base = trimesh.creation.box(extents=[base_size, base_size, float(thickness_mm)])
        base.apply_translation([0.0, 0.0, float(thickness_mm) / 2.0])

    elif mode == "hex":
        radius = base_size / 2.0
        base = trimesh.creation.cylinder(radius=radius, height=float(thickness_mm), sections=int(hex_sides))
        base.apply_translation([0.0, 0.0, float(thickness_mm) / 2.0])

    else:
        raise ValueError(f"Unknown base_mode: {base_mode}")

    return trimesh.util.concatenate([mesh_up, base])


def _enforce_allowed_size(size_mm: float) -> float:
    allowed = {80, 120, 180, 200}
    s = int(round(float(size_mm)))
    if s not in allowed:
        raise ValueError(f"size_mm must be one of {sorted(allowed)} (got {size_mm})")
    return float(s)


# =========================
# Main pipeline: GLB -> STL (with options)
# =========================
def generate_final_stl(
    input_glb: str,
    output_stl: str,
    size_mm: float = 120,
    base_mode: str = "circle",          # none|circle|oval|square|hex
    base_thickness_mm: float = 5.0,
    orient_opt: bool = True,
    support_minimize: bool = True,
    max_overhang_deg: float = 45.0,
) -> dict:
    """
    Full pipeline:
    GLB -> OBJ (Blender) -> mesh (trimesh) -> (orient) -> scale to height -> ground -> base -> STL

    Returns metadata useful for debugging / UI.
    """
    size_mm = _enforce_allowed_size(size_mm)

    output_stl_p = Path(output_stl).resolve()
    output_stl_p.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        obj_path = tmp / "model.obj"

        # 1) GLB -> OBJ
        glb_to_obj_blender(input_glb, str(obj_path))

        # 2) Load OBJ (skip materials to avoid PIL dependency)
        mesh = trimesh.load(
            str(obj_path),
            force="mesh",
            skip_materials=True,
            process=False,
        )
        if mesh.is_empty:
            raise RuntimeError("Loaded mesh is empty")

        # 3) Convert to mm (assume source is meters-ish; this keeps your pipeline consistent)
        #    If later you confirm Tripo outputs already in meters, keep 1000.
        mesh.apply_scale(1000.0)

        # 4) Orientation optimization
        orient_meta = {"note": "orient_opt=false"}
        if orient_opt:
            if support_minimize:
                orient_meta = orient_for_print_support_min(
                    mesh,
                    max_overhang_deg=max_overhang_deg,
                    w_overhang=1.0,
                    w_stability=0.35,
                    max_poses=12,
                )
            else:
                orient_meta = orient_for_print_simple(mesh)

        # 5) Ground first (stability for size measurement)
        drop_to_ground(mesh)

        # 6) Scale to target height (mm)
        scale_to_height_mm(mesh, size_mm)

        # 7) Ground again (after scaling)
        drop_to_ground(mesh)

        # 8) Base
        if base_mode and base_mode.lower() not in ("none", "off", "false", "0"):
            mesh = add_base_shape(
                mesh,
                base_mode=base_mode,
                thickness_mm=base_thickness_mm,
                size_ratio=0.75,
                margin_mm=2.0,
                oval_xy_ratio=1.35,
                hex_sides=6,
            )
            # After base, ensure ground is correct
            drop_to_ground(mesh)

        # 9) Export STL
        mesh.export(str(output_stl_p))

    meta = {
        "output_stl": str(output_stl_p),
        "size_mm": float(size_mm),
        "base_mode": str(base_mode),
        "base_thickness_mm": float(base_thickness_mm),
        "orient_opt": bool(orient_opt),
        "support_minimize": bool(support_minimize),
        "max_overhang_deg": float(max_overhang_deg),
        "orient_meta": orient_meta,
    }
    print("[OK] STL generated:", meta["output_stl"])
    return meta