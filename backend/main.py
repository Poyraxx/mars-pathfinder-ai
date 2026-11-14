import asyncio
import io
import json
import math
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import data
from .coordinates import point_to_cell, cell_to_point
from .events import EventLog
from .navigation import cost_grid, plan_legs, metrics, inflate_obstacles
from .perception import infer, localize_obstacles, hazard_overlay
from .schemas import MissionRequest, ObservationRequest, PlanRequest

LAYERS = {
 "viking":{"name":"Viking color mosaic","kind":"Optical imagery","layer":"Mars_Viking_MDIM21_ClrMosaic_global_232m","ext":"jpg","resolution_m":232,"coverage":[-180,-90,180,90],"mission":"Viking · NASA / USGS"},
 "mola":{"name":"MOLA elevation / hillshade","kind":"Topography visualization","layer":"Mars_MGS_MOLA_ClrShade_merge_global_463m","ext":"jpg","resolution_m":463,"coverage":[-180,-90,180,90],"mission":"Mars Global Surveyor · NASA / USGS"},
 "themis":{"name":"THEMIS daytime infrared","kind":"Thermal context","layer":"THEMIS_DayIR_ControlledMosaics_100m_v2_oct2018","ext":"png","resolution_m":100,"coverage":[-180,-65,180,65],"mission":"Mars Odyssey · NASA / ASU / USGS"},
}
SOURCES = [
 {"name":"NASA Mars Trek","url":"https://trek.nasa.gov/mars/","usage":"Challenge resource; optical, topographic and thermal tile layers"},
 {"name":"HiRISE MSL landing-site DTM","url":"https://hirise.lpl.arizona.edu/dtm/ESP_023957_1755","usage":"Numerical DEM; NASA/JPL-Caltech/University of Arizona/USGS"},
 {"name":"MSL Navcam RDR","url":"https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=MSL-M-NAVCAM-5-RDR-V1.0","usage":"Archived camera reference images, XYZ and error products"},
 {"name":"MSL PLACES","url":"https://planetarydata.jpl.nasa.gov/w10n/msl/msl_places/","usage":"Archive localization and coordinate frame transforms"},
 {"name":"AI4Mars v0.6 · CC BY 4.0","url":"https://zenodo.org/records/15995036","usage":"Surface terrain labels; Swan, Ono & Atha (2022)"},
 {"name":"MOLA / HRSC 200m DEM","url":"https://astro.arcgis.com/arcgis/rest/services/OnMars/MDEM200M/ImageServer","usage":"Open global 3D terrain, hosted by Esri; NASA/USGS/ESA/DLR/FU Berlin"},
]
missions = OrderedDict()
overlays = {}
events = EventLog()
compute_lock = asyncio.Lock()
http = None
tile_cache = OrderedDict()
BOOT_ID = uuid.uuid4().hex[:12]


@asynccontextmanager
async def lifespan(app):
    global http
    data.load_regions()
    http = httpx.AsyncClient(timeout=35, follow_redirects=True, limits=httpx.Limits(max_connections=8))
    yield
    await http.aclose()


app = FastAPI(title="Mars Pathfinder AI",version="0.1.0",lifespan=lifespan)


def mission_or_404(mission_id):
    if mission_id not in missions:
        raise HTTPException(404,detail={"code":"mission_not_found","message":"Session expired or unknown mission. Create a new mission."})
    return missions[mission_id]


def reject(code,message,status=422):
    raise HTTPException(status,detail={"code":code,"message":message})


def route_result(mission,mode):
    region = data.regions[mission["request"]["region_id"]]
    request = mission["request"]
    cost = cost_grid(region.analysis,mode,request["profile"]["max_slope_deg"], mission["blocked"] if mode=="surface" else None)
    points = [point_to_cell(p,region.meta) for p in [request["start"],*request["stops"],request["goal"]]]
    began = time.perf_counter()
    try:
        path,total = plan_legs(cost,points,region.meta["cell_m"])
    except (ValueError,TimeoutError) as exc:
        return {"mode":mode,"status":str(exc),"points":[],"map_version":mission["map_version"]}
    stats = metrics(path,region.elevation,region.analysis["slope"],region.meta["cell_m"],request["profile"]["speed_m_s"])
    # Retain all path cells for validation; decimate only the visualization payload.
    draw_indices = np.unique(np.linspace(0,len(path)-1,min(len(path),400)).astype(int))
    return {"mode":mode,"status":"ready","map_version":mission["map_version"],
            "planning_ms":round((time.perf_counter()-began)*1000,1),"cost":round(total,2),
            **stats,"points":[cell_to_point(*path[i],region.meta) for i in draw_indices],
            "cells":[list(p) for p in path]}


@app.get("/healthz")
def health(): return {"status":"ok","boot_id":BOOT_ID,"regions":list(data.regions),"model_ready":data.model_status().get("ready",False)}


@app.get("/api/catalog")
def catalog():
    return {"body":499,"latitude_type":"planetocentric","longitude_domain":"[-180,180)",
            "layers":[{"id":key,**val} for key,val in LAYERS.items()],"sources":SOURCES,
            "regions":[r.meta for r in data.regions.values()],"model":data.model_status(),
            "frames":data.frame_catalog(),"boot_id":BOOT_ID,
            "disclaimer":"Research prototype. Not flight qualified or certified for human EVA safety.",
            "orbital_method":"Deterministic DEM analysis; learned orbital model not installed"}


@app.get("/api/regions/{region_id}")
def region_info(region_id: str):
    if region_id not in data.regions: reject("unsupported_region","No validated planning DEM is installed for this region.",404)
    r = data.regions[region_id]
    return {**r.meta,"shape":list(r.elevation.shape),"valid_fraction":round(float(r.analysis["valid"].mean()),4)}


@app.post("/api/missions",status_code=201)
def create_mission(request: MissionRequest):
    if request.region_id not in data.regions: reject("insufficient_dem","Choose an installed numerical DEM planning area.")
    region = data.regions[request.region_id]
    for p in [request.start,*request.stops,request.goal]:
        row,col = point_to_cell(p.model_dump(),region.meta)
        if not (0<=row<region.elevation.shape[0] and 0<=col<region.elevation.shape[1]) or not region.analysis["valid"][row,col]:
            reject("outside_coverage","A waypoint lies outside valid DEM coverage. Pick inside the outlined planning area.")
    if len(missions)>=64: reject("queue_full","Session capacity reached. Retry after server restart.",503)
    mid = uuid.uuid4().hex
    missions[mid] = {"id":mid,"request":request.model_dump(),"map_version":0,"routes":[],"observations":[],
                     "blocked":np.zeros(region.elevation.shape,dtype=bool),"processed":set(),"state":"created"}
    events.append(mid,"mission.created","Mission created · Human EVA research profile",map_version=0)
    return {"id":mid,"map_version":0}


@app.post("/api/missions/{mission_id}/plan",status_code=202)
async def plan(mission_id: str, payload: PlanRequest):
    m = mission_or_404(mission_id)
    if payload.expected_map_version is not None and payload.expected_map_version!=m["map_version"]:
        reject("stale_map_version","Refresh mission state before planning.",409)
    if compute_lock.locked(): reject("queue_full","Another computation is active. Please retry shortly.",429)
    await compute_lock.acquire()
    job_id = uuid.uuid4().hex
    m["state"]="planning"
    async def work():
        try:
            version = m["map_version"]
            events.append(mission_id,"analysis.started","Orbital · Computing slope, roughness and terrain costs",map_version=version)
            routes = []
            try:
                for mode in ["distance","terrain","surface"]:
                    routes.append(await asyncio.to_thread(route_result,m,mode))
                if version==m["map_version"]:
                    m["routes"] = routes
                    m["state"] = "ready"
                    events.append(mission_id,"route.planned","Navigation · Route comparison calculated",map_version=version,job_id=job_id)
            except Exception:
                m["state"]="error"
                events.append(mission_id,"job.failed","Planning failed; no route was asserted",job_id=job_id)
        finally:
            compute_lock.release()
    asyncio.create_task(work())
    return {"job_id":job_id}


@app.post("/api/missions/{mission_id}/observations",status_code=202)
async def observe(mission_id: str,payload: ObservationRequest):
    m = mission_or_404(mission_id)
    if payload.base_map_version!=m["map_version"]: reject("stale_map_version","Observation based on an older map.",409)
    if payload.frame_id in m["processed"]: reject("duplicate_observation","This source frame has already been processed for this mission.",409)
    frame = next((f for f in data.frame_catalog() if f["id"]==payload.frame_id),None)
    if frame is None: reject("unknown_frame","Choose a catalogued reference frame.")
    if frame.get("region_id") != m["request"]["region_id"]: reject("wrong_region","Camera frame and mission region do not match.")
    if not data.model_status().get("ready"): reject("model_not_ready","A trained Surface checkpoint has not been installed.",503)
    if compute_lock.locked(): reject("queue_full","Another computation is active. Retry shortly.",429)
    await compute_lock.acquire()
    job_id=uuid.uuid4().hex
    m["state"]="perceiving"
    async def work():
        try:
            events.append(mission_id,"perception.started","Surface · Running learned terrain segmentation",frame_id=payload.frame_id,job_id=job_id)
            try:
                frame,mask,conf,png,result = await asyncio.to_thread(infer,payload.frame_id)
                overlays[(mission_id,payload.frame_id)]=png
                events.append(mission_id,"perception.completed","Surface · Segmentation completed on archived camera image",**result)
                loc=await asyncio.to_thread(localize_obstacles,frame,mask,conf,data.regions[m["request"]["region_id"]].meta,
                                            m["request"]["profile"].get("obstacle_step_m",.15))
                if loc.get("hazard_mask") is not None and loc["hazard_mask"].any():
                    overlays[(mission_id,payload.frame_id)]=await asyncio.to_thread(hazard_overlay,png,loc["hazard_mask"])
                result["localization"]={k:v for k,v in loc.items() if k not in ("points_rc","hazard_mask")}
                if loc["accepted"] and "hazard_pixels" in loc:
                    events.append(mission_id,"geometry.measured",
                                  f"Stereo geometry · {loc['hazard_pixels']} px stand >= {loc['step_threshold_m']:.2f} m above local ground",
                                  frame_id=payload.frame_id,hazard_components=loc["hazard_components"],max_step_m=loc["max_step_m"],
                                  registration_error_m=round(loc["registration_error_m"],2),model_big_rock_pixels_not_used=loc["model_big_rock_pixels_not_used"])
                m["observations"].append(result)
                m["processed"].add(payload.frame_id)
                if not loc["accepted"]:
                    events.append(mission_id,"localization.rejected",loc["reason"],frame_id=payload.frame_id,map_version=m["map_version"])
                else:
                    region=data.regions[m["request"]["region_id"]]
                    inflation=(m["request"]["profile"]["clearance_m"]+loc["registration_error_m"])/region.meta["cell_m"]
                    update=inflate_obstacles(region.elevation.shape,loc["points_rc"],inflation)
                    changed=bool(np.any(update & ~m["blocked"]))
                    if changed:
                        old_route = next((r for r in m["routes"] if r["mode"]=="surface"),None)
                        m["previous_route"]=old_route
                        m["blocked"] |= update
                        m["map_version"]+=1
                        m["routes"]=[]
                        events.append(mission_id,"map.updated","Navigation · Grounded obstacle footprints added; replanning",map_version=m["map_version"])
                        m["routes"]=[await asyncio.to_thread(route_result,m,mode) for mode in ["distance","terrain","surface"]]
                        events.append(mission_id,"route.replanned","Navigation · Updated route calculated from the revised map",map_version=m["map_version"])
                    else: events.append(mission_id,"route.unchanged","Navigation · No new supported obstacle cells",map_version=m["map_version"])
                m["state"]="ready"
            except Exception as exc:
                m["state"]="error"
                events.append(mission_id,"job.failed",f"Surface processing failed: {type(exc).__name__}. Map was not updated.",job_id=job_id)
        finally:
            compute_lock.release()
    asyncio.create_task(work())
    return {"job_id":job_id}


def public_state(m):
    return {k:v for k,v in m.items() if k not in ("blocked","processed")}


@app.get("/api/missions/{mission_id}/state")
def state(mission_id: str): return public_state(mission_or_404(mission_id))


@app.get("/api/missions/{mission_id}/report")
def report(mission_id: str):
    m=mission_or_404(mission_id)
    return {**public_state(m),"events":events.read(mission_id),"sources":SOURCES,"model":data.model_status(),
            "disclaimer":"Actual computation on archived data. No EVA safety certification. Session storage is ephemeral."}


@app.get("/api/missions/{mission_id}/events")
async def event_stream(mission_id: str,request: Request,after: int=0):
    mission_or_404(mission_id)
    try: after=max(after,int(request.headers.get("last-event-id","0")))
    except ValueError: reject("invalid_cursor","Invalid SSE sequence cursor.")
    async def stream():
        cursor=after
        while not await request.is_disconnected():
            for event in events.read(mission_id,cursor):
                cursor=event["sequence_id"]
                yield f"id: {cursor}\ndata: {json.dumps(event)}\n\n"
            yield ": heartbeat\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(stream(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@app.get("/api/frames")
def frames(region_id: str|None=None): return [f for f in data.frame_catalog() if region_id is None or f["region_id"]==region_id]


@app.get("/api/context/orbital.png")
def orbital_context():
    path=data.DATA_DIR/"pds"/"orbital-crop.png"
    if not path.exists(): reject("context_not_ready","Orbital context image not prepared.",404)
    return FileResponse(path)


@app.get("/api/frames/{frame_id}/image")
def frame_image(frame_id: str):
    f=next((f for f in data.frame_catalog() if f["id"]==frame_id),None)
    if not f: reject("unknown_frame","Unknown reference frame.",404)
    return FileResponse(data.DATA_DIR / f["image_path"])


@app.get("/api/missions/{mission_id}/overlays/{frame_id}")
def overlay(mission_id:str,frame_id:str):
    mission_or_404(mission_id)
    png=overlays.get((mission_id,frame_id))
    if png is None: reject("not_processed","No model result exists for this frame.",404)
    return Response(png,media_type="image/png")


@app.get("/api/regions/{region_id}/risk.png")
def risk_image(region_id:str):
    if region_id not in data.regions: reject("unsupported_region","Unknown planning region.",404)
    r=data.regions[region_id]
    slope=r.analysis["slope"]
    risk=np.nan_to_num(np.clip(slope/20,0,1))
    rgba=np.stack([60+195*risk,180-110*risk,160-100*risk,np.where(r.analysis["valid"],150,0)],axis=-1).astype(np.uint8)
    buffer=io.BytesIO(); Image.fromarray(rgba).save(buffer,format="PNG")
    return Response(buffer.getvalue(),media_type="image/png")


@app.get("/api/regions/{region_id}/heightmap")
def heightmap(region_id:str):
    if region_id not in data.regions: reject("unsupported_region","Unknown region.",404)
    region=data.regions[region_id]
    return Response(np.asarray(region.elevation,dtype='<f4').tobytes(),media_type='application/octet-stream',headers={'Cache-Control':'public,max-age=86400'})


@app.get("/api/regions/{region_id}/ortho.png")
def region_ortho(region_id:str):
    if region_id not in data.regions: reject("unsupported_region","Unknown region.",404)
    path=data.DATA_DIR/'pds'/'region-ortho.png'
    if not path.exists(): reject("context_not_ready","Regional ortho not prepared.",404)
    return FileResponse(path)


async def tile_response(key,url,media):
    if key in tile_cache: return Response(tile_cache[key],media_type=media,headers={"Cache-Control":"public,max-age=86400"})
    try:
        async with http.stream("GET",url) as result:
            result.raise_for_status()
            chunks=[]; total=0
            async for chunk in result.aiter_bytes():
                total+=len(chunk)
                if total>6_000_000: reject("upstream_tile_too_large","Tile limit exceeded.",502)
                chunks.append(chunk)
        payload=b"".join(chunks)
    except httpx.HTTPError: reject("tile_unavailable","Scientific tile service is temporarily unavailable.",502)
    tile_cache[key]=payload
    while len(tile_cache)>128 or sum(map(len,tile_cache.values()))>32*1024*1024: tile_cache.popitem(last=False)
    return Response(payload,media_type=media,headers={"Cache-Control":"public,max-age=86400"})


@app.get("/api/tiles/{layer}/{z}/{y}/{x}")
async def imagery_tile(layer:str,z:int,y:int,x:int):
    if layer not in LAYERS or not 0<=z<=(9 if layer=="themis" else 7) or not 0<=x<2**(z+1) or not 0<=y<2**z:
        reject("invalid_tile","Invalid planetary tile.",404)
    l=LAYERS[layer]
    url=f"https://trek.nasa.gov/tiles/Mars/EQ/{l['layer']}/1.0.0/default/default028mm/{z}/{y}/{x}.{l['ext']}"
    return await tile_response(f"{layer}/{z}/{y}/{x}",url,"image/png" if l["ext"]=="png" else "image/jpeg")


@app.get("/api/terrain/{z}/{y}/{x}")
async def elevation_tile(z:int,y:int,x:int):
    if not 0<=z<=7 or not 0<=x<2**(z+1) or not 0<=y<2**z: reject("invalid_tile","Invalid elevation tile.",404)
    return await tile_response(f"dem/{z}/{y}/{x}",f"https://astro.arcgis.com/arcgis/rest/services/OnMars/MDEM200M/ImageServer/tile/{z}/{y}/{x}","application/octet-stream")


if (data.ROOT / "dist").exists():
    app.mount("/",StaticFiles(directory=data.ROOT/"dist",html=True),name="frontend")
