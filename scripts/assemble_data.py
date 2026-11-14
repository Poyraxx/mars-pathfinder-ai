"""Build the app catalogue from downloaded, traced PDS assets; never certify registration."""
import json
import shutil
from pathlib import Path
import numpy as np
from backend.coordinates import cell_to_point

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data"/"prepared"


def main():
    pds=OUT/"pds"
    source=json.loads((pds/"pds-candidate.json").read_text("utf-8"))
    dem=json.loads((pds/"dem-crop.json").read_text("utf-8"))
    target=OUT/"gale";target.mkdir(parents=True,exist_ok=True)
    if not (target/"elevation.npy").exists():
        shutil.copyfile(pds/"dem-crop.npy",target/"elevation.npy")
    z=np.load(target/"elevation.npy")
    n=z.shape[0]
    meta={"id":"gale","name":"Gale · Rocknest","description":"Curiosity Sol57 · real HiRISE PLACES orbital DEM",
          "body":499,"latitude_type":"planetocentric","radius_m":dem["sphere_radius_m"],"standard_parallel_deg":0,
          "central_lon_deg":0,"central_lat_deg":0,"west_m":dem["upper_left_easting_m"],"north_m":dem["upper_left_northing_m"],
          "cell_m":dem["pixel_resolution_m"],"shape":list(z.shape),"source_id":dem["source_lid"]+"::"+dem["version"],
          "source_resolution_m":1.0,"planning_resolution_m":1.0,"source_url":dem["image_url"],"source_page":"https://planetarydata.jpl.nasa.gov/w10n/msl/msl_places/",
          "elevation_datum":dem["height_reference"],"source_provenance":dem,"attribution":"NASA/JPL-Caltech/USGS/UArizona · MSL PLACES",
          "processing":"Native PDS float32 DEM subset. No synthetic data or hole filling.","coverage_level":"research_topographic","registration_verified":False,"display_surface_height_m":float(np.nanmedian(z))}
    corners=[cell_to_point(r,c,meta) for r,c in [(0,0),(0,n-1),(n-1,n-1),(n-1,0)]]
    meta["bounds"]=[min(p["lon"] for p in corners),min(p["lat"] for p in corners),max(p["lon"] for p in corners),max(p["lat"] for p in corners)]
    meta["center"]=cell_to_point(n//2,n//2,meta)
    meta["default_start"]=cell_to_point(300,170,meta)
    meta["default_goal"]=cell_to_point(200,330,meta)
    ortho_meta=OUT/'pds'/'region-ortho.json'
    if ortho_meta.exists():
        meta['ortho_bounds']=json.loads(ortho_meta.read_text('utf-8'))['render_rectangle_degrees_west_south_east_north']
    (target/"metadata.json").write_text(json.dumps(meta,indent=2),"utf-8")
    frames=[]
    for candidate in source["frames"]:
        f={**candidate,"region_id":"gale","image_path":"pds/"+candidate["reference_image"],"xyz_path":"pds/"+candidate["xyz_file"],"xye_path":"pds/"+candidate["xye_file"],
           "registration":{"verified":False,"horizontal_error_bound_m":None,"evidence":None,"reason":candidate["registration_status"]},
           "source_url":candidate["products"][0]["label_url"]}
        frames.append(f)
    (OUT/"frames.json").write_text(json.dumps(frames,indent=2),"utf-8")
    print(f"Prepared real {n}x{n} 1m DEM and {len(frames)} matched PDS camera products. Independent registration remains false.")


if __name__=="__main__":main()
