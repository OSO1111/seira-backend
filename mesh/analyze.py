import sys
import trimesh
import json

mesh = trimesh.load(sys.argv[1])

result = {
    "is_watertight": mesh.is_watertight,
    "faces": len(mesh.faces),
    "vertices": len(mesh.vertices),
    "components": len(mesh.split())
}

print(json.dumps(result))
