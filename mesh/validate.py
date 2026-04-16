import sys
import json
from pathlib import Path

import trimesh


def main():
    if len(sys.argv) < 2:
        raise RuntimeError("usage: python mesh/validate.py <input_path>")

    input_path = sys.argv[1]
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

    components = mesh.split(only_watertight=False)
    component_count = len(components)

    printable = bool(
        (not mesh.is_empty)
        and (mesh.faces is not None and len(mesh.faces) > 0)
        and (mesh.vertices is not None and len(mesh.vertices) > 0)
    )

    # 현재는 최소 기준으로 두고, watertight면 더 높은 confidence로 보게끔
    status = "printable" if printable else "failed"

    result = {
        "filename": Path(input_path).name,
        "status": status,
        "printable": printable,
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "faces": int(len(mesh.faces)) if mesh.faces is not None else 0,
        "vertices": int(len(mesh.vertices)) if mesh.vertices is not None else 0,
        "components": int(component_count),
        "bounds": {
            "min": mesh.bounds[0].tolist() if mesh.bounds is not None else [0, 0, 0],
            "max": mesh.bounds[1].tolist() if mesh.bounds is not None else [0, 0, 0],
            "extents": mesh.bounding_box.extents.tolist() if mesh.bounding_box is not None else [0, 0, 0],
        },
        "volume": float(mesh.volume) if mesh.is_volume else 0.0,
        "area": float(mesh.area) if mesh.area is not None else 0.0,
    }

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
