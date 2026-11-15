"""Prepare a bounded numerical HiRISE DEM crop, preserving source provenance."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
RAW=Path(os.environ.get("LOCALAPPDATA",ROOT/"data"/"raw"))/"mars-pathfinder"/"raw"
URL="https://hirise.lpl.arizona.edu/PDS/DTM/ESP/ORB_023900_023999/ESP_023957_1755_ESP_024023_1755/DTEEC_023957_1755_024023_1755_U01.IMG"


def main():
    RAW.mkdir(parents=True,exist_ok=True)
    src_path=RAW/"DTEEC_023957_1755_024023_1755_U01.IMG"
    if not src_path.exists():
        print("Downloading real HiRISE landing-site DEM (~458 MB)",flush=True)
        urllib.request.urlretrieve(URL,src_path.with_suffix(".part"))
        src_path.with_suffix(".part").replace(src_path)
    out=ROOT/"data"/"prepared"/"gale"; out.mkdir(parents=True,exist_ok=True)
    radius=3396190.0
    lat,lon=-4.58999,137.4484
    standard_parallel=lat
    cx=radius*np.cos(np.radians(standard_parallel))*np.radians(lon)
    cy=radius*np.radians(lat)
    cell=2.; n=512
    target_crs=f"+proj=eqc +lat_ts={standard_parallel} +lat_0=0 +lon_0=0 +a={radius} +b={radius} +units=m +no_defs"
    with rasterio.open(src_path) as src:
        print({"shape":src.shape,"crs":str(src.crs),"nodata":src.nodata,"transform":str(src.transform)},flush=True)
        for n in [512,384,256,128,64]:
            west,north=cx-n*cell/2,cy+n*cell/2
            target=np.full((n,n),np.nan,dtype=np.float32)
            reproject(source=rasterio.band(src,1),destination=target,src_transform=src.transform,src_crs=src.crs,
                      src_nodata=src.nodata,dst_transform=from_origin(west,north,cell,cell),dst_crs=target_crs,
                      dst_nodata=np.nan,resampling=Resampling.bilinear)
            print({"cells":n,"valid_fraction":float(np.isfinite(target).mean()),"center":float(target[n//2,n//2])},flush=True)
            if np.isfinite(target).mean()>=.9: break
        source_res=list(src.res)
    print({"crop_valid_fraction":float(np.isfinite(target).mean())},flush=True)
    if np.isfinite(target).mean()<.9: raise RuntimeError("DEM crop has insufficient valid coverage")
    np.save(out/"elevation.npy",target)
    meta={"id":"gale","name":"Gale · Rocknest","description":"Curiosity early traverse · numerical HiRISE DEM planning area",
          "body":499,"latitude_type":"planetocentric","radius_m":radius,"standard_parallel_deg":standard_parallel,
          "central_lon_deg":0,"central_lat_deg":0,"west_m":west,"north_m":north,"cell_m":cell,
          "shape":[n,n],"center":{"lat":lat,"lon":lon},"elevation_datum":"areoid (source DTEEC)",
          "source_id":"DTEEC_023957_1755_024023_1755_U01","source_resolution_m":source_res,
          "planning_resolution_m":cell,"source_url":URL,"source_page":"https://hirise.lpl.arizona.edu/dtm/ESP_023957_1755",
          "source_sha256":hashlib.file_digest(src_path.open("rb"),"sha256").hexdigest(),
          "attribution":"NASA/JPL-Caltech/UArizona/USGS","processing":"Bilinear resampling of real DTM to 2m local standard-parallel eqc grid; no synthetic relief.",
          "coverage_level":"research_topographic","registration_verified":False}
    from backend.coordinates import cell_to_point
    corners=[cell_to_point(r,c,meta) for r,c in [(0,0),(0,n-1),(n-1,n-1),(n-1,0)]]
    meta["bounds"]=[min(p["lon"] for p in corners),min(p["lat"] for p in corners),max(p["lon"] for p in corners),max(p["lat"] for p in corners)]
    meta["default_start"]=cell_to_point(int(n*.6),int(n*.37),meta)
    meta["default_goal"]=cell_to_point(int(n*.43),int(n*.62),meta)
    (out/"metadata.json").write_text(json.dumps(meta,indent=2),"utf-8")
    print(json.dumps({"ready":True,"valid_fraction":float(np.isfinite(target).mean()),"min":float(np.nanmin(target)),"max":float(np.nanmax(target))}),flush=True)


if __name__=="__main__": main()
