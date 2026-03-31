from __future__ import annotations

import os
import subprocess
from pathlib import Path

import trimesh


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _basic_repair(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.fix_normals()
    return mesh


def _load_as_single_mesh(input_path: str) -> trimesh.Trimesh:
    scene_or_mesh = trimesh.load(input_path, force="scene")

    if isinstance(scene_or_mesh, trimesh.Scene):
        if not scene_or_mesh.geometry:
            raise ValueError("No geometry found in model file")

        meshes = []
        for geom in scene_or_mesh.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)

        if not meshes:
            raise ValueError("Scene has no mesh geometry")

        combined = trimesh.util.concatenate(meshes)
        return combined

    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh

    raise ValueError("Unsupported mesh type")


def convert_model(input_path: str, output_path: str, output_format: str) -> str:
    output_format = output_format.lower()
    _ensure_parent(output_path)

    if output_format == "glb":
        mesh = _load_as_single_mesh(input_path)
        mesh = _basic_repair(mesh)
        mesh.export(output_path, file_type="glb")
        return output_path

    if output_format in {"stl", "obj"}:
        mesh = _load_as_single_mesh(input_path)
        mesh = _basic_repair(mesh)
        mesh.export(output_path, file_type=output_format)
        return output_path

    if output_format == "fbx":
        blender_path = os.environ.get("BLENDER_PATH", "blender")
        script = f'''
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=r"{input_path}")
bpy.ops.export_scene.fbx(filepath=r"{output_path}", use_selection=False)
'''
        result = subprocess.run(
            [blender_path, "--background", "--python-expr", script],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "FBX conversion failed. "
                f"stdout={result.stdout}\n"
                f"stderr={result.stderr}"
            )
        return output_path

    raise ValueError(f"Unsupported output_format: {output_format}")
