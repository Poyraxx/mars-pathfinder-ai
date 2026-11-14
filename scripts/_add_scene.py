"""Append/replace one reviewed scene in the v6 superpixel annotation file (helper for review sessions)."""
import json,sys
from pathlib import Path
path=Path('data/annotations/hazcam-operational-v6-superpixels.json')
doc=json.loads(path.read_text('utf-8')) if path.exists() else {"version":6,"classes":{"0":"ground_or_small_stone","1":"substantial_obstacle","2":"rover_hardware","3":"sky_or_distant_unknown","255":"unknown_or_unreviewed"},
 "method":"See data/annotations/operational-v6-review-protocol.json. Superpixel ids refer to scripts/label_superpixels.py PARAMS on the pinned native image; polygons are native 1024 px coordinates and override superpixels.","scenes":[]}
scene=json.loads(sys.stdin.read())
doc['scenes']=[s for s in doc['scenes'] if s['id']!=scene['id']]+[scene]
path.write_text(json.dumps(doc,indent=1),'utf-8');print(len(doc['scenes']),'scenes')
