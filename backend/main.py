import asyncio
import threading
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
from .global_dem import GlobalRegion
from .camera_route import project_route, view_mission
from . import orbital
from . import features
from . import replay
from . import local_navigation
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
 {"name":"MSL Front Hazcam RDR","url":"https://planetarydata.jpl.nasa.gov/img/data/msl/MSLHAZ_1XXX/DATA/","usage":"Actual forward-facing imagery, matched stereo XYZ/XYE and per-frame CAHVORE calibration"},
 {"name":"NASA-AMMOS VICAR camera models","url":"https://github.com/NASA-AMMOS/VICAR/blob/master/vos/p2/sub/cahvor/cmod_cahvore.c","usage":"Reference CAHVORE camera projection; Apache-2.0, Caltech/Todd Litwin"},
 {"name":"MSL PLACES","url":"https://planetarydata.jpl.nasa.gov/w10n/msl/msl_places/","usage":"Archive localization and coordinate frame transforms"},
 {"name":"AI4Mars v0.6 · CC BY 4.0","url":"https://zenodo.org/records/15995036","usage":"Surface terrain labels; Swan, Ono & Atha (2022)"},
 {"name":"HiRISE labeled landmarks v3.2 · CC BY 4.0","url":"https://zenodo.org/records/4002935","usage":"Orbital landmark training data; Wagstaff et al., NASA JPL"},
 {"name":"Gazetteer of Planetary Nomenclature","url":"https://planetarynames.wr.usgs.gov/","usage":"IAU-approved Mars feature names, types, diameters and centres (USGS Astrogeology)"},
 {"name":"MOLA / HRSC 200m DEM","url":"https://astro.arcgis.com/arcgis/rest/services/OnMars/MDEM200M/ImageServer","usage":"Open global 3D terrain, hosted by Esri; NASA/USGS/ESA/DLR/FU Berlin"},
]
missions = OrderedDict()
overlays = {}
events = EventLog()
compute_lock = asyncio.Lock()
local_navigation_lock = asyncio.Lock()
http = None
tile_cache = OrderedDict()
BOOT_ID = uuid.uuid4().hex[:12]
MAX_GLOBAL_MISSIONS = 6


@asynccontextmanager
async def lifespan(app):
    global http
    data.load_regions()
    threading.Thread(target=_warm_default_scenario, daemon=True).start()
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


def region_of(mission):
    return mission.get("_region") or data.regions[mission["request"]["region_id"]]


def route_result(mission,mode):
    region = region_of(mission)
    request = mission["request"]
    blocked = mission["blocked"] if mode=="surface" else None
    avoid = mission.get("_avoid")
    if avoid is not None and mode!="distance":
        blocked = avoid if blocked is None else (blocked | avoid)
    cost = cost_grid(region.analysis,mode,request["profile"]["max_slope_deg"], blocked)
    points = [point_to_cell(p,region.meta) for p in [request["start"],*request["stops"],request["goal"]]]
    began = time.perf_counter()
    try:
        path,total = plan_legs(cost,points,region.meta["cell_m"])
    except (ValueError,TimeoutError) as exc:
        return {"mode":mode,"status":str(exc),"points":[],"map_version":mission["map_version"]}
    stats = metrics(path,region.elevation,region.analysis["slope"],region.meta["cell_m"],request["profile"]["speed_m_s"])
    # Retain all path cells for validation; decimate only the visualization payload.
    draw_indices = np.unique(np.linspace(0,len(path)-1,min(len(path),400)).astype(int))
    drawn=[cell_to_point(*path[i],region.meta) for i in draw_indices]
    return {"mode":mode,"status":"ready","map_version":mission["map_version"],
            "planning_ms":round((time.perf_counter()-began)*1000,1),"cost":round(total,2),
            **stats,"points":drawn,"features":features.along_route(drawn),
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
            "orbital_method":"Deterministic DEM terrain costs + learned HiRISE landmark classifier (context)" if orbital.card().get("ready") else "Deterministic DEM analysis; learned orbital model not installed",
            "orbital_model":{k:v for k,v in orbital.card().items() if k in ("ready","name","classes","test","dataset","dataset_url","dataset_license","limitations","split_counts","ctx_transfer_check")}}


@app.get("/api/regions/{region_id}")
def region_info(region_id: str):
    if region_id not in data.regions: reject("unsupported_region","No validated planning DEM is installed for this region.",404)
    r = data.regions[region_id]
    return {**r.meta,"shape":list(r.elevation.shape),"valid_fraction":round(float(r.analysis["valid"].mean()),4)}


@app.post("/api/missions",status_code=201)
async def create_mission(request: MissionRequest):
    points=[p.model_dump() for p in [request.start,*request.stops,request.goal]]
    if request.region_id=="global":
        try:
            region=await asyncio.to_thread(GlobalRegion,points)
        except ValueError as exc:
            messages={"polar_not_supported":"Global coarse planning is limited to latitudes within +-80 degrees.",
}
            reject(str(exc),messages.get(str(exc),"Global DEM could not be prepared."))
        except Exception as exc:
            reject("dem_unavailable",f"Global DEM could not be prepared ({type(exc).__name__}: {str(exc)[:160]}).",502)
    elif request.region_id in data.regions:
        region = data.regions[request.region_id]
    else:
        reject("insufficient_dem","Choose an installed numerical DEM planning area.")
    for p in points:
        row,col = point_to_cell(p,region.meta)
        if not (0<=row<region.elevation.shape[0] and 0<=col<region.elevation.shape[1]) or not region.analysis["valid"][row,col]:
            reject("outside_coverage","A waypoint lies outside valid DEM coverage. Pick inside the outlined planning area.")
    avoid=None; avoided=[]
    if request.avoid_features:
        try:
            avoid,avoided=features.avoid_mask(region.meta,region.elevation.shape,request.avoid_features)
        except KeyError as exc:
            reject("unknown_feature",f"No IAU-named feature called {exc.args[0]!r}.")
        for p in points:
            row,col=point_to_cell(p,region.meta)
            if avoid[row,col]: reject("waypoint_in_avoided_feature","A waypoint lies inside a feature you asked to avoid.")
    if len(missions)>=64: reject("queue_full","Session capacity reached. Retry after server restart.",503)
    global_ids=[k for k,v in missions.items() if v.get("_region") is not None]
    while len(global_ids)>=MAX_GLOBAL_MISSIONS:
        missions.pop(global_ids.pop(0),None)
    mid = uuid.uuid4().hex
    missions[mid] = {"id":mid,"request":request.model_dump(),"map_version":0,"routes":[],"observations":[],
                     "blocked":np.zeros(region.elevation.shape,dtype=bool),"processed":set(),"state":"created",
                     "region":{k:v for k,v in region.meta.items() if k!="shape"}|{"shape":list(region.elevation.shape)}}
    if avoid is not None:
        missions[mid]["_avoid"]=avoid; missions[mid]["avoided_features"]=avoided
        events.append(mid,"features.avoided","Avoiding "+", ".join(f"{a['name']} ({a['type']}, {a['diameter_km']:.0f} km)" for a in avoided)+" · circular IAU footprint",map_version=0)
    if request.region_id=="global":
        missions[mid]["_region"]=region
        events.append(mid,"mission.created",f"Mission created · global coarse DEM {region.meta['cell_m']:.0f} m cells (level {region.meta['source_level']})",map_version=0)
    else:
        events.append(mid,"mission.created","Mission created · Human EVA research profile",map_version=0)
    return {"id":mid,"map_version":0,"region":missions[mid]["region"]}


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
                loc=await asyncio.to_thread(localize_obstacles,frame,mask,conf,region_of(m).meta,
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
                    region=region_of(m)
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
    return {k:v for k,v in m.items() if k not in ("blocked","processed","_region","_avoid")}


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


@app.get("/api/frames/{frame_id}/view-mission")
def frame_view_mission(frame_id: str):
    f=next((f for f in data.frame_catalog() if f["id"]==frame_id),None)
    if not f: reject("unknown_frame","Unknown reference frame.",404)
    if not f["registration"].get("verified"): reject("unregistered_frame","This frame has no accepted orbital registration.",409)
    return view_mission(f,data.regions[f["region_id"]].meta)


@app.get("/api/missions/{mission_id}/camera-route/{frame_id}")
def camera_route(mission_id: str, frame_id: str, mode: str="surface", previous: bool=False):
    m=mission_or_404(mission_id)
    f=next((f for f in data.frame_catalog() if f["id"]==frame_id),None)
    if not f: reject("unknown_frame","Unknown reference frame.",404)
    if f.get("region_id")!=m["request"]["region_id"]: reject("wrong_region","Camera frame and mission region do not match.")
    reg=f["registration"]
    if not reg.get("verified"): reject("unregistered_frame","Route cannot be placed in an image without accepted orbital registration.",409)
    route=m.get("previous_route") if previous else next((r for r in m["routes"] if r["mode"]==mode and r["status"]=="ready"),None)
    if route is None or route.get("status")!="ready": reject("no_route","No planned route of this kind exists for the mission.",409)
    tolerance=max(0.35,min(1.0,reg["horizontal_error_bound_m"]/2))
    return {**project_route(f,region_of(m).meta,route["cells"],tolerance),"mode":mode,"map_version":route["map_version"],
            "registration_error_m":reg["horizontal_error_bound_m"]}


@app.get("/api/frames/{frame_id}/display")
def frame_display(frame_id: str):
    f=next((f for f in data.frame_catalog() if f["id"]==frame_id),None)
    if not f: reject("unknown_frame","Unknown reference frame.",404)
    return FileResponse(data.DATA_DIR/(f.get("display_path") or f["image_path"]),headers={"Cache-Control":"public,max-age=86400"})


@app.get('/api/frames/{frame_id}/local-navigation')
async def frame_local_navigation(frame_id: str, goal_lat: float|None=None, goal_lon: float|None=None):
    if (goal_lat is None) != (goal_lon is None):
        reject('incomplete_goal','Both goal coordinates are required.')
    if goal_lat is not None and (not math.isfinite(goal_lat) or not math.isfinite(goal_lon) or not -90<=goal_lat<=90 or not -180<=goal_lon<180):
        reject('invalid_goal','Use finite Mars latitude/longitude coordinates.')
    if not any(f['id']==frame_id for f in data.frame_catalog()):
        reject('unknown_frame','Unknown reference frame.',404)
    goal=None if goal_lat is None else {'lat':goal_lat,'lon':goal_lon}
    key=(frame_id,None if goal is None else (round(goal_lat,7),round(goal_lon,7)))
    if key in frame_local_cache: return frame_local_cache[key]
    # A frame already computed for a whole replay is served from that batch, never re-scanned.
    for payload in list(scenario_local_cache.values()):
        g=payload['goal']
        if key[1] is not None and (round(g['lat'],7),round(g['lon'],7))==key[1] and frame_id in payload['frames']:
            return payload['frames'][frame_id]
    if local_navigation_lock.locked():
        reject('queue_full','Local stereo processing is active; retry shortly.',429)
    async with local_navigation_lock:
        try:
            result=await asyncio.to_thread(local_navigation.compute,frame_id,goal)
            frame_local_cache[key]=result
            while len(frame_local_cache)>600: frame_local_cache.pop(next(iter(frame_local_cache)))
            return result
        except (OSError,ValueError,KeyError) as exc:
            return {'frame_id':frame_id,'status':'unavailable','reason':f'Local stereo/model inputs unavailable ({type(exc).__name__}); no route asserted.',
                    'segments':[],'ribbons':[],'is_ground_projection':True}


orbital_results={}
frame_local_cache={}
scenario_local_cache={}
scenario_local_pending={}
scenario_local_guard=threading.Lock()


replay_pending={}


def _replay_once(scenario_id, goal):
    """replay.compute with concurrent callers for the same scenario/goal sharing one computation.

    Replays are read-only and cached in replay._cache, so they do not take the global compute lock:
    a slow first computation on a free-tier CPU must never block other users' requests."""
    key=(scenario_id,None if goal is None else (round(goal['lat'],7),round(goal['lon'],7)))
    with scenario_local_guard:
        event=replay_pending.get(key); owner=event is None
        if owner: event=threading.Event(); replay_pending[key]=event
    if not owner:
        event.wait()
    try:
        return replay.compute(scenario_id,None,goal)
    finally:
        if owner:
            with scenario_local_guard: replay_pending.pop(key,None)
            event.set()


def _scenario_local(scenario_id, goal):
    """Local stereo routes for every frame of a replay, computed once on this server and kept in memory."""
    key=(scenario_id,None if goal is None else (round(goal['lat'],7),round(goal['lon'],7)))
    with scenario_local_guard:
        if key in scenario_local_cache: return scenario_local_cache[key]
        event=scenario_local_pending.get(key)
        owner=event is None
        if owner:
            event=threading.Event(); scenario_local_pending[key]=event
    if not owner:
        event.wait(); return scenario_local_cache.get(key)
    try:
        began=time.perf_counter()
        result=_replay_once(scenario_id,goal)
        frames={}
        for st in result['steps']:
            if st['frame_id'] in frames: continue
            try:
                frames[st['frame_id']]=local_navigation.compute(st['frame_id'],result['goal'])
            except (OSError,ValueError,KeyError) as exc:
                frames[st['frame_id']]={'frame_id':st['frame_id'],'status':'unavailable','reason':f'Local stereo/model inputs unavailable ({type(exc).__name__}).','segments':[],'ribbons':[],'is_ground_projection':True}
        payload={'scenario':scenario_id,'goal':result['goal'],'frames':frames,'compute_ms':round((time.perf_counter()-began)*1000,1),
                 'computed_at':datetime.now(timezone.utc).isoformat(timespec='seconds'),
                 'note':'Computed on this server from the archived stereo products and kept in memory for smooth playback.'}
        with scenario_local_guard:
            scenario_local_cache[key]=payload
            while len(scenario_local_cache)>6: scenario_local_cache.pop(next(iter(scenario_local_cache)))
        return payload
    finally:
        with scenario_local_guard:
            scenario_local_pending.pop(key,None)
        event.set()


def _warm_default_scenario():
    try:
        first=next(iter(replay.scenarios()),None)
        if first: _scenario_local(first['id'],None)
    except Exception:
        pass


@app.get('/api/scenarios/{scenario_id}/local-navigation')
async def scenario_local_navigation(scenario_id: str, goal_lat: float|None=None, goal_lon: float|None=None):
    if not any(s['id']==scenario_id for s in replay.scenarios()): reject('unknown_scenario','Unknown scenario.',404)
    if (goal_lat is None)!=(goal_lon is None): reject('incomplete_goal','Both goal coordinates are required.')
    goal=None if goal_lat is None else {'lat':goal_lat,'lon':goal_lon}
    if goal and (not math.isfinite(goal_lat) or not math.isfinite(goal_lon)): reject('invalid_goal','Use finite coordinates.')
    try:
        return await asyncio.to_thread(_scenario_local,scenario_id,goal)
    except ValueError as exc:
        reject(str(exc),'Pick a destination inside the scenario planning area.')


@app.post("/api/regions/{region_id}/orbital-analysis")
async def orbital_analysis(region_id: str):
    if region_id not in data.regions: reject("unsupported_region","Unknown region.",404)
    if not orbital.card().get("ready"): reject("orbital_model_not_ready","The learned orbital model is not installed.",503)
    if compute_lock.locked(): reject("queue_full","Another computation is active. Retry shortly.",429)
    async with compute_lock:
        result,png=await asyncio.to_thread(orbital.analyze_region,region_id)
    orbital_results[region_id]=png
    return result


@app.post("/api/missions/{mission_id}/orbital-analysis")
async def mission_orbital_analysis(mission_id: str, mode: str="surface"):
    m=mission_or_404(mission_id)
    if not orbital.card().get("ready"): reject("orbital_model_not_ready","The learned orbital model is not installed.",503)
    route=next((r for r in m["routes"] if r["mode"]==mode and r["status"]=="ready"),None)
    if route is None: reject("no_route","Plan the mission before running the orbital analysis.",409)
    if compute_lock.locked(): reject("queue_full","Another computation is active. Retry shortly.",429)
    async with compute_lock:
        events.append(mission_id,"orbital.started","Orbital · Classifying CTX windows along the route",map_version=m["map_version"])
        try:
            result=await asyncio.to_thread(orbital.analyze_route,route["points"],route["distance_m"])
        except Exception as exc:
            events.append(mission_id,"job.failed",f"Orbital analysis failed: {type(exc).__name__}")
            reject("orbital_failed","Orbital imagery or model unavailable.",502)
    found=", ".join(f"{k.replace('_',' ')} {v}" for k,v in result["class_counts"].items() if v)
    events.append(mission_id,"orbital.completed",f"Orbital · {len(result['windows'])} CTX windows of {result['window_m']:.0f} m: {found or 'no coverage'}",map_version=m["map_version"])
    m["orbital"]=result
    return result


@app.get("/api/regions/{region_id}/orbital-analysis.png")
def orbital_png(region_id: str):
    if region_id not in orbital_results: reject("not_processed","Run the orbital analysis first.",404)
    return Response(orbital_results[region_id],media_type="image/png")


@app.get("/api/scenarios")
def list_scenarios():
    out=[]
    for sc in replay.scenarios():
        frames={f["id"]:f for f in data.frame_catalog()}
        fs=[frames[i] for i in sc["frames"] if i in frames]
        out.append({k:sc[k] for k in ("id","title","subtitle","story","references","note","camera","sols","region_id")}|
                   {"frame_count":len(fs),"registered_frames":sum(bool(f["registration"].get("verified")) for f in fs),"first_frame":fs[0]["id"] if fs else None})
    return out


@app.get("/api/scenarios/{scenario_id}/replay")
async def scenario_replay(scenario_id: str, goal_lat: float|None=None, goal_lon: float|None=None):
    if not any(s["id"]==scenario_id for s in replay.scenarios()): reject("unknown_scenario","Unknown scenario.",404)
    goal=None if goal_lat is None or goal_lon is None else {"lat":goal_lat,"lon":goal_lon}
    try:
        result=await asyncio.to_thread(_replay_once,scenario_id,goal)
    except ValueError as exc:
        reject(str(exc),"Pick a destination inside the scenario's 1 m planning area.")
    return {**result,'scenario':{**result['scenario'],'demo_available':(data.DATA_DIR/'demos'/f'{scenario_id}.mp4').is_file() and goal is None}}


@app.get('/api/scenarios/{scenario_id}/demo.mp4')
def scenario_demo(scenario_id:str):
    if not any(s['id']==scenario_id for s in replay.scenarios()):
        reject('unknown_scenario','Unknown scenario.',404)
    path=data.DATA_DIR/'demos'/f'{scenario_id}.mp4'
    if not path.is_file():
        reject('demo_not_prepared','No exported film is available for this scenario.',404)
    return FileResponse(path,media_type='video/mp4',filename=f'mars-pathfinder-{scenario_id}.mp4')


@app.get("/api/nomenclature")
def nomenclature():
    path=data.DATA_DIR/"global"/"nomenclature.json"
    if not path.exists(): reject("not_prepared","Nomenclature catalogue not prepared.",404)
    return FileResponse(path,media_type="application/json",headers={"Cache-Control":"public,max-age=86400"})


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
