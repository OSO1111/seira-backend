import sys
import json
from pathlib import Path

import trimesh


def main():
    if len(sys.argv) < 2:
        raise RuntimeError("usage: python mesh/analyze.py <input_path>")

    input_path = sys.argv[1]
    mesh = trimesh.load(input_path, force="mesh")

    if mesh is None:
        raise RuntimeError("failed to load mesh")

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("loaded object is not a mesh")

    components = mesh.split(only_watertight=False)
    component_count = len(components)

    bbox_extents = mesh.bounding_box.extents.tolist() if mesh.bounding_box is not None else [0, 0, 0]

    issues = []

    if not mesh.is_watertight:
        issues.append("not_watertight")

    if component_count > 1:
        issues.append("multiple_components")

    if not mesh.is_winding_consistent:
        issues.append("inconsistent_winding")

    if mesh.faces is None or len(mesh.faces) == 0:
        issues.append("no_faces")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        issues.append("no_vertices")

    # trimesh에서 직접 hole 개수를 신뢰성 있게 주진 않아서 추정성 정보로 둠
    estimated_hole_risk = not mesh.is_watertight

    result = {
        "filename": Path(input_path).name,
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "is_empty": bool(mesh.is_empty),
        "faces": int(len(mesh.faces)) if mesh.faces is not None else 0,
        "vertices": int(len(mesh.vertices)) if mesh.vertices is not None else 0,
        "components": int(component_count),
        "bounds": {
            "min": mesh.bounds[0].tolist() if mesh.bounds is not None else [0, 0, 0],
            "max": mesh.bounds[1].tolist() if mesh.bounds is not None else [0, 0, 0],
            "extents": bbox_extents,
        },
        "volume": float(mesh.volume) if mesh.is_volume else 0.0,
        "area": float(mesh.area) if mesh.area is not None else 0.0,
        "estimated_hole_risk": bool(estimated_hole_risk),
        "issues": issues,
    }

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
