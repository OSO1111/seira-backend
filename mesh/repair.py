import sys
import json
from pathlib import Path

import trimesh


ALLOWED_STEPS = {
    "remove_duplicate_faces",
    "remove_degenerate_faces",
    "keep_largest_component",
    "merge_vertices",
    "close_holes",
    "manifold_rebuild",
    "voxel_remesh",
}


def load_mesh(input_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(input_path, force="mesh")

    if mesh is None:
        raise RuntimeError("failed to load mesh")

    if isinstance(mesh, trimesh.Scene):
        geometry = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geometry:
            raise RuntimeError("scene has no mesh geometry")
        mesh = trimesh.util.concatenate(geometry)

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("loaded object is not a mesh")

    return mesh


def keep_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    return max(parts, key=lambda m: len(m.faces))


def merge_vertices(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.merge_vertices()
    return mesh


def close_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    try:
        mesh.fill_holes()
    except Exception:
        pass
    return mesh


def manifold_rebuild(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    # trimesh만으로 가능한 범위의 기본 정리
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        mesh.remove_infinite_values()
    except Exception:
        pass

    try:
        mesh.process(validate=True)
    except Exception:
        pass

    return mesh


def voxel_remesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    # 너무 공격적이라 최소 fallback으로만 사용
    # bounding box 기반으로 대략 voxel pitch 계산
    extents = mesh.bounding_box.extents
    max_extent = max(extents) if extents is not None and len(extents) > 0 else 1.0
    pitch = max(max_extent / 64.0, 1e-4)

    try:
        voxel = mesh.voxelized(pitch=pitch)
        remeshed = voxel.marching_cubes
        if isinstance(remeshed, trimesh.Trimesh) and len(remeshed.faces) > 0:
            return remeshed
    except Exception:
        pass

    return mesh


def apply_plan(mesh: trimesh.Trimesh, plan: list[str]) -> trimesh.Trimesh:
    for step in plan:
        if step not in ALLOWED_STEPS:
            continue

        if step == "remove_duplicate_faces":
            try:
                mesh.remove_duplicate_faces()
            except Exception:
                pass

        elif step == "remove_degenerate_faces":
            try:
                mesh.remove_degenerate_faces()
            except Exception:
                pass

        elif step == "keep_largest_component":
            mesh = keep_largest_component(mesh)

        elif step == "merge_vertices":
            mesh = merge_vertices(mesh)

        elif step == "close_holes":
            mesh = close_holes(mesh)

        elif step == "manifold_rebuild":
            mesh = manifold_rebuild(mesh)

        elif step == "voxel_remesh":
            mesh = voxel_remesh(mesh)

    return mesh


def export_mesh(mesh: trimesh.Trimesh, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    suffix = out.suffix.lower()
    if suffix not in {".glb", ".stl", ".obj", ".ply", ".off"}:
        raise RuntimeError(f"unsupported output format: {suffix}")

    mesh.export(output_path)


def main():
    if len(sys.argv) < 4:
        raise RuntimeError(
            "usage: python mesh/repair.py <input_path> <output_path> <repair_plan_json>"
        )

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    plan_json = sys.argv[3]

    try:
        plan = json.loads(plan_json)
        if not isinstance(plan, list):
            raise RuntimeError("repair_plan_json must be a JSON array")
    except json.JSONDecodeError as e:
        raise RuntimeError("invalid repair_plan_json") from e

    mesh = load_mesh(input_path)

    # 기본 정리 1회
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        mesh.remove_infinite_values()
    except Exception:
        pass

    # 계획 적용
    mesh = apply_plan(mesh, plan)

    # 마무리 기본 정리
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        mesh.process(validate=True)
    except Exception:
        pass

    if mesh.faces is None or len(mesh.faces) == 0:
        raise RuntimeError("repair result has no faces")

    export_mesh(mesh, output_path)


if __name__ == "__main__":
    main()
