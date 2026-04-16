import sys
import trimesh
import json

input_path = sys.argv[1]
output_path = sys.argv[2]
plan = json.loads(sys.argv[3])

mesh = trimesh.load(input_path)

# 기본 정리
mesh.remove_duplicate_faces()
mesh.remove_degenerate_faces()

# 가장 큰 덩어리만 남기기
components = mesh.split()
if len(components) > 1:
    mesh = max(components, key=lambda m: len(m.faces))

mesh.export(output_path)
