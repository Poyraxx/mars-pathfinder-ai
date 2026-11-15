import {useEffect,useMemo,useRef,useState} from 'react';
import {Eye,Navigation,Pause,Play,SkipBack,SkipForward,Video} from 'lucide-react';
import {api,useMission} from './store';
import type {Frame} from './types';

type Projection={segments:number[][][];visible_route_m:number;route_length_m:number;tolerance_m:number;camera_ground_range_m:[number,number];map_version:number};

export default function CameraView({dual}:{dual:boolean}){
 const s=useMission();
 const frame=s.catalog?.frames.find(f=>f.id===s.frameId);
 const observation=s.mission?[...s.mission.observations].reverse().find(o=>o.frame_id===s.frameId):undefined;
 const [mask,setMask]=useState(false),[playing,setPlaying]=useState(false);
 const [route,setRoute]=useState<Projection|null>(null),[oldRoute,setOldRoute]=useState<Projection|null>(null),[note,setNote]=useState('');
 const sequence=useMemo(()=>(s.catalog?.frames??[]).filter(f=>(f.sequence??'drive')===(frame?.sequence??'drive')).sort((a,b)=>(a.sequence_index??0)-(b.sequence_index??0)),[s.catalog,frame?.sequence]);
 const index=sequence.findIndex(f=>f.id===s.frameId);
 const timer=useRef<number|undefined>(undefined);
 const step=(d:number)=>{if(sequence.length)s.set({frameId:sequence[(index+d+sequence.length)%sequence.length].id});};
 useEffect(()=>{if(!playing)return;timer.current=window.setInterval(()=>{const st=useMission.getState();const seq=(st.catalog?.frames??[]).filter(f=>(f.sequence??'drive')===(frame?.sequence??'drive')).sort((a,b)=>(a.sequence_index??0)-(b.sequence_index??0));const i=seq.findIndex(f=>f.id===st.frameId);if(seq.length)st.set({frameId:seq[(i+1)%seq.length].id});},1400);return()=>window.clearInterval(timer.current);},[playing,frame?.sequence]);
 const routeKey=`${s.mission?.id}|${s.mission?.map_version}|${s.mission?.routes.length}|${s.frameId}|${s.activeRoute}`;
 useEffect(()=>{
  setRoute(null);setOldRoute(null);setNote('');
  const m=s.mission;if(!m||!frame||!m.routes.some(r=>r.status==='ready'))return;
  if(!frame.registration.verified){setNote('This frame has no accepted orbital registration, so the route cannot be placed in it.');return;}
  if(m.region&&m.region.id!==frame.region_id){setNote('This mission is outside the camera region.');return;}
  let live=true;
  fetch(`/api/missions/${m.id}/camera-route/${frame.id}?mode=${s.activeRoute}`).then(r=>r.ok?r.json():r.json().then(e=>Promise.reject(new Error(e.detail?.message??'Route projection unavailable')))).then((p:Projection)=>{if(live){setRoute(p);if(!p.segments.length)setNote('The planned route is not visible from this camera position.');}}).catch(e=>live&&setNote(String(e.message??e)));
  if(m.previous_route)fetch(`/api/missions/${m.id}/camera-route/${frame.id}?previous=true`).then(r=>r.ok?r.json():null).then(p=>{if(live&&p)setOldRoute(p);}).catch(()=>{});
  return()=>{live=false;};
 // eslint-disable-next-line react-hooks/exhaustive-deps
 },[routeKey]);
 async function planHere(f:Frame){
  const region=s.catalog?.regions.find(r=>r.id===f.region_id);if(!region)return;
  try{
   const vm=await api<{start:{lat:number;lon:number};goal:{lat:number;lon:number};goal_range_m:number}>(`/api/frames/${f.id}/view-mission`);
   s.set({region,planMode:'local',start:vm.start,goal:vm.goal,stops:[],busy:true,error:null,events:[],mission:null});
   const created=await api<{id:string;map_version:number}>('/api/missions',{region_id:region.id,start:vm.start,goal:vm.goal,stops:[]});
   s.set({mission:await api(`/api/missions/${created.id}/state`)});
   await api(`/api/missions/${created.id}/plan`,{expected_map_version:created.map_version});
  }catch(error){s.set({busy:false,error:error instanceof Error?error.message:String(error)});}
 }
 const draw=(p:Projection|null,cls:string)=>p?.segments.map((seg,i)=><polyline key={cls+i} className={cls} points={seg.map(q=>`${q[0]},${q[1]}`).join(' ')}/>);
 const src=frame?(observation&&mask&&s.mission?`/api/missions/${s.mission.id}/overlays/${frame.id}`:`/api/frames/${frame.id}/display`):'';
 return <div className={`camera-main ${dual?'dual':''}`}>
  <div className="camera-caption"><Video size={15}/><span>ARCHIVED ROVER VIEWPOINT · {frame?.sequence==='panorama'?'ROCKNEST PANORAMA':'DRIVE SEQUENCE'}</span><span className="live-tag">{route?.segments.length?'ROUTE FROM STEREO GEOMETRY':observation?'MODEL OUTPUT':'SOURCE IMAGE'}</span></div>
  {frame?<div className="camera-frame"><div className="camera-stage"><img src={src} alt="Real MSL Navcam archive frame"/>
   <svg className="camera-route" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Planned route projected into the camera image">{draw(oldRoute,'old')}{draw(route,'new')}</svg>
   {route?.segments.length?<div className="camera-route-legend"><span className="new"/>Planned route · {route.visible_route_m.toFixed(1)} m visible of {route.route_length_m.toFixed(1)} m{oldRoute?.segments.length?<><span className="old"/>Before camera update</>:null}</div>:null}
  </div></div>:<div className="empty-camera"><Video size={36}/><p>Preparing real Navcam assets</p></div>}
  {note&&<div className="camera-note">{note}</div>}
  <div className="camera-main-footer">
   <span>MSL · SOL {frame?.sol??'—'} · {frame?.acquired_at?.slice(11,19)??''} UTC · {index+1}/{sequence.length}{frame?.instrument_azimuth_deg!=null?` · AZ ${frame.instrument_azimuth_deg.toFixed(0)}°`:''}</span>
   <div className="camera-controls">
    <button onClick={()=>step(-1)} aria-label="Previous frame"><SkipBack size={14}/></button>
    <button onClick={()=>setPlaying(!playing)} aria-label={playing?'Pause sequence':'Play sequence'}>{playing?<Pause size={14}/>:<Play size={14}/>}</button>
    <button onClick={()=>step(1)} aria-label="Next frame"><SkipForward size={14}/></button>
    <button onClick={()=>frame&&planHere(frame)} disabled={!frame?.registration.verified||s.busy} title="Plan from the rover position toward the centre of this view"><Navigation size={14}/>Plan in this view</button>
    <button onClick={()=>setMask(!mask)} disabled={!observation}><Eye size={14}/>{mask?'Show source':'Show mask'}</button>
   </div>
  </div>
 </div>;
}
