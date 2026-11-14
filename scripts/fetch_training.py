"""Fetch a bounded original-publisher subset with grouped, reproducible splits."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from remote_zip import open_ai4mars


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--count",type=int,default=400); parser.add_argument("--inspect",action="store_true")
    args=parser.parse_args()
    target=Path(os.environ.get("LOCALAPPDATA","."))/"mars-pathfinder"/"training"
    target.mkdir(parents=True,exist_ok=True)
    z=open_ai4mars(); names=z.namelist(); prefix="ai4mars-dataset-merged-0.6/"
    info=z.read(prefix+"info.md").decode("utf-8",errors="replace")
    (target/"info.md").write_text(info,"utf-8")
    if args.inspect:
        print(info[:22000]); print("\n".join(n for n in names if "/msl/" in n and 'mcam' not in n and not n.endswith('/'))[:7000]); return
    label_names=[n for n in names if "/msl/" in n and "/labels/train/" in n and 'mcam' not in n and n.lower().endswith(".png")]
    image_names={Path(n).stem:n for n in names if "/msl/" in n and "/images/" in n and n.lower().endswith((".jpg",".jpeg",".png"))}
    if not label_names: raise RuntimeError("Unexpected publisher archive layout; run --inspect")
    entries=[]; groups={}; cap={"train":args.count,"validation":max(40,args.count//5),"test":max(40,args.count//5)}
    for name in sorted(label_names):
        stem=Path(name).stem.removesuffix('_merged')
        image=image_names.get(stem)
        if not image: continue
        match=re.search(r"N[LR][A-Z]?_(\d{9})",stem)
        if not match: continue
        sclk=int(match[1])
        if abs(sclk-402556045)<88775*3: continue  # Exclude demo and neighbouring sols entirely.
        group=str(sclk//(88775*3))
        bucket=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%100
        split="train" if bucket<70 else "validation" if bucket<85 else "test"
        if sum(e["split"]==split for e in entries)>=cap[split]: continue
        # Limit adjacent frames from one acquisition group.
        if groups.get(group,0)>=12: continue
        groups[group]=groups.get(group,0)+1
        entries.append({"image_member":image,"label_member":name,"group":group,"split":split,"stem":stem})
        if all(sum(e["split"]==s for e in entries)>=cap[s] for s in cap): break
    if not entries: raise RuntimeError("No matching MSL/Navcam labels and images")
    # Read in archive byte order to reduce repeated HTTP range requests.
    needed={n for e in entries for n in [e["image_member"],e["label_member"]]}
    for i,n in enumerate(sorted(needed,key=lambda n:z.getinfo(n).header_offset)):
        local=target/"members"/n.removeprefix(prefix)
        local.parent.mkdir(parents=True,exist_ok=True)
        if not local.exists(): local.write_bytes(z.read(n))
        if i%25==0: print(f"Fetched {i}/{len(needed)} archive members",flush=True)
    for e in entries:
        e["image"]=str(target/"members"/e["image_member"].removeprefix(prefix))
        e["label"]=str(target/"members"/e["label_member"].removeprefix(prefix))
    manifest={"source":"https://zenodo.org/records/15995036","license":"CC-BY-4.0","version":"0.6",
              "split_method":"3-sol acquisition-time blocks, deterministic hash; max12 images/block; demo +-3sol excluded",
              "note":"Test is a held-out crowdsourced-label subset, not an expert gold benchmark.","entries":entries}
    (target/"manifest.json").write_text(json.dumps(manifest,indent=2),"utf-8")
    print({s:sum(e["split"]==s for e in entries) for s in cap},flush=True)


if __name__=="__main__": main()
