import sys
import trimesh
import json

mesh = trimesh.load(sys.argv[1])

result = {
    "printable": mesh.is_watertight,
    "faces": len(mesh.faces)
}

print(json.dumps(result))
