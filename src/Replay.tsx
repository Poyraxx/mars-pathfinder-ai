import {useEffect,useMemo,useRef,useState} from 'react';
import * as C from 'cesium';
import {ArrowUpRight,Flag,LoaderCircle,MapPin,Pause,Play,RotateCcw,SkipBack,SkipForward,Video,Box} from 'lucide-react';
import {MARS,ROVER_MODEL,demHeightAt,ensureLocalDEM,scheme,terrain} from './Globe';
import {api,useMission} from './store';
import type {ReplayResult,Scenario} from './types';

const EVENT_TEXT:Record<string,string>={off_route:'Off route · recalculating',u_turn:'Make a U-turn · route behind the rover',obstacle:'Obstacle ahead · re-routing',no_alternative:'No alternative route found'};
let replayRequest=0;

export async function loadReplay(id:string,goal?:{lat:number;lon:number}|null){
 const s=useMission.getState();
 const request=++replayRequest;
 s.set({busy:true,error:null});
 try{
  const q=goal?`?goal_lat=${goal.lat}&goal_lon=${goal.lon}`:'';
  const r=await api<ReplayResult>(`/api/scenarios/${id}/replay${q}`);
  if(request===replayRequest)useMission.getState().set({replay:r,replayStep:0,busy:false,replayView:'navcam',replayImageReady:null,frameId:r.steps[0]?.frame_id??s.frameId});
 }catch(error){if(request===replayRequest)useMission.getState().set({busy:false,error:error instanceof Error?error.message:String(error)});}
}

/** Sidebar content for Replay mode: ready-made scenarios built from real archived drives. */
export function ReplayPanel(){
 const s=useMission();
 const [selected,setSelected]=useState<string|null>(null);
 useEffect(()=>{if(!s.scenarios.length)void api<Scenario[]>('/api/scenarios').then(list=>{const st=useMission.getState();st.set({scenarios:list});if(!st.replay&&list[0])void loadReplay(list[0].id);}).catch(error=>s.set({error:String(error)}));},[s.scenarios.length]);
 useEffect(()=>{if(!s.replay)return;for(const step of s.replay.steps.slice(s.replayStep,s.replayStep+3)){const img=new Image();img.src=`/api/frames/${step.frame_id}/display`;}},[s.replay,s.replayStep]);
 useEffect(()=>{const onGoal=(e:Event)=>{const id=useMission.getState().replay?.scenario.id;if(id)void loadReplay(id,(e as CustomEvent).detail);};window.addEventListener('replay-goal',onGoal);return()=>window.removeEventListener('replay-goal',onGoal);},[]);
 const current=s.replay?.scenario.id??selected;
 return <div className="panel-section">
  <h2 className="panel-title">Rover navigation</h2>
  <p className="panel-lede">Real NASA camera frames on the left. The same remaining route and recorded rover position on the right.</p>
  <div className="scenario-list">{s.scenarios.map(sc=><button key={sc.id} className={`scenario ${current===sc.id?'active':''}`} onClick={()=>{setSelected(sc.id);void loadReplay(sc.id);}} disabled={s.busy}>
   {sc.first_frame&&<img src={`/api/frames/${sc.first_frame}/image`} alt=""/>}
   <span><strong>{sc.title}</strong><small>{sc.subtitle}</small><small>{sc.frame_count} real archive frames</small></span>
  </button>)}</div>
  {s.replay&&<div className="scenario-detail">
   <p>{s.replay.scenario.story}</p>
   <div className="button-row">
    <button className="ghost-button" onClick={()=>s.set({replayGoalPick:true,pick:'goal'})} disabled={s.busy}><Flag size={14}/>{s.replayGoalPick?'Click the map…':'Set a new destination'}</button>
    {s.replay.goal.custom&&<button className="ghost-button" onClick={()=>void loadReplay(s.replay!.scenario.id)} disabled={s.busy}><RotateCcw size={14}/>Real destination</button>}
   </div>
   {s.replay.scenario.demo_available&&<a className="ghost-button demo-download" href={`/api/scenarios/${s.replay.scenario.id}/demo.mp4`} download><Video size={14}/>Download 30-second demo</a>}
   <h3>Sources</h3>
   <ul className="reference-list">{s.replay.scenario.references.map(r=><li key={r.url}><a href={r.url} target="_blank" rel="noreferrer">{r.label}<ArrowUpRight size={12}/></a></li>)}</ul>
   <p className="fine-print">{s.replay.scenario.note} Computed live in {Math.round(s.replay.compute_ms)} ms · re-plan when off route by more than {s.replay.parameters.deviation_m} m.</p>
  </div>}
 </div>;
}

/** Left half in Replay mode: first-person 3D view from the real rover pose, or the archived Navcam frame. */
export function ReplayView(){
 const s=useMission();
 const r=s.replay,st=r?.steps[Math.min(s.replayStep,(r?.steps.length??1)-1)];
 return <section className="replay-view" aria-label="Rover view">
  <div className="replay-view-tabs" role="tablist">
   <button role="tab" aria-selected={s.replayView==='navcam'} className={s.replayView==='navcam'?'active':''} onClick={()=>s.set({replayView:'navcam'})}><Video size={14}/>Real camera</button>
   <button role="tab" aria-selected={s.replayView==='3d'} className={s.replayView==='3d'?'active':''} onClick={()=>s.set({replayView:'3d'})}><Box size={14}/>3D simulation</button>
  </div>
  {!r?<div className="replay-empty"><MapPin size={22}/><p>Choose a drive replay to start.</p></div>:s.replayView==='3d'?<FirstPerson/>:<NavcamFrame/>}
  {s.busy&&<div className="replay-loading" role="status"><LoaderCircle className="spin" size={22}/>Calculating the shared route…</div>}
  {st?.event&&<div className={`replay-toast ${st.event}`} role="status">{EVENT_TEXT[st.event]}<small>{(st.deviation_before_replan_m??st.deviation_m).toFixed(1)} m before update · route v{st.route_version}</small></div>}
  {r&&st&&<div className="replay-readout"><span><b>{st.remaining_m.toFixed(1)} m</b> to destination</span><span>Sol {st.sol} · {st.time.slice(11,19)} UTC</span><span>{st.deviation_m.toFixed(1)} m off route</span></div>}
 </section>;
}

function FirstPerson(){
 const mount=useRef<HTMLDivElement>(null),viewer=useRef<C.Viewer|null>(null);
 const {replay,replayStep,catalog}=useMission();
 const [ready,setReady]=useState(false);
 useEffect(()=>{
  let disposed=false;
  void ensureLocalDEM().then(()=>{
   if(disposed||!mount.current)return;
   const v=new C.Viewer(mount.current,{animation:false,timeline:false,baseLayerPicker:false,geocoder:false,sceneModePicker:false,homeButton:false,fullscreenButton:false,navigationHelpButton:false,selectionIndicator:false,infoBox:false,baseLayer:false,
    terrainProvider:terrain(),globe:new C.Globe(MARS),skyAtmosphere:false,requestRenderMode:true,maximumRenderTimeChange:Infinity});
   v.scene.backgroundColor=C.Color.fromCssColorString('#c9a27c');
   v.scene.globe.baseColor=C.Color.fromCssColorString('#8a5a40');
   v.scene.globe.depthTestAgainstTerrain=true;
   v.scene.globe.maximumScreenSpaceError=1.5;
   (v.camera.frustum as C.PerspectiveFrustum).fov=C.Math.toRadians(70);
   v.scene.screenSpaceCameraController.enableInputs=false;
   v.imageryLayers.addImageryProvider(new C.UrlTemplateImageryProvider({url:'/api/tiles/viking/{z}/{y}/{x}',tilingScheme:scheme,maximumLevel:7}));
   const region=catalog?.regions[0];
   if(region?.ortho_bounds){const [w,so,e,n]=region.ortho_bounds;C.SingleTileImageryProvider.fromUrl(`/api/regions/${region.id}/ortho.png`,{rectangle:C.Rectangle.fromDegrees(w,so,e,n),ellipsoid:MARS}).then(p=>{if(!v.isDestroyed()){v.imageryLayers.addImageryProvider(p);v.scene.requestRender();}});}
   viewer.current=v;setReady(true);
  });
  return()=>{disposed=true;viewer.current?.destroy();viewer.current=null;};
 // eslint-disable-next-line react-hooks/exhaustive-deps
 },[]);
 useEffect(()=>{
  const v=viewer.current;if(!v||!replay)return;
  const st=replay.steps[Math.min(replayStep,replay.steps.length-1)];
  v.entities.removeAll();
  const ground=demHeightAt(st.rover.lat,st.rover.lon)??-4517;
  if(st.route.length>1)v.entities.add({polyline:{positions:C.Cartesian3.fromDegreesArray([st.rover.lon,st.rover.lat,...st.route.flatMap(p=>[p.lon,p.lat])],MARS),width:9,material:new C.PolylineGlowMaterialProperty({glowPower:.25,color:C.Color.fromCssColorString(st.event==='u_turn'?'#f2a65a':'#3ce1ff')}),clampToGround:true}});
  const gh=demHeightAt(replay.goal.lat,replay.goal.lon)??ground;
  v.entities.add({position:C.Cartesian3.fromDegrees(replay.goal.lon,replay.goal.lat,gh+1.2,MARS),point:{pixelSize:14,color:C.Color.fromCssColorString('#f2a65a'),outlineColor:C.Color.WHITE,outlineWidth:2,disableDepthTestDistance:Infinity},label:{text:'DESTINATION',font:'600 12px sans-serif',fillColor:C.Color.WHITE,outlineColor:C.Color.BLACK,outlineWidth:3,style:C.LabelStyle.FILL_AND_OUTLINE,pixelOffset:new C.Cartesian2(0,-20),disableDepthTestDistance:Infinity}});
  // Navcam height on Curiosity's mast is about 1.9 m above the ground.
  v.camera.flyTo({destination:C.Cartesian3.fromDegrees(st.rover.lon,st.rover.lat,ground+1.9,MARS),orientation:{heading:C.Math.toRadians(st.rover.yaw_deg),pitch:C.Math.toRadians(-14),roll:0},duration:.7,easingFunction:C.EasingFunction.LINEAR_NONE});
  v.scene.requestRender();
 },[replay,replayStep,ready]);
 return <div className="first-person"><div ref={mount} className="first-person-canvas"/><div className="view-caption">Rendered from HiRISE 1 m DEM and 25 cm orthophoto at the rover's PLACES pose · not a camera image</div></div>;
}

function NavcamFrame(){
 const {replay,replayStep,replayImageReady,set,catalog}=useMission();
 const st=replay!.steps[Math.min(replayStep,replay!.steps.length-1)];
 const cr=st.camera_route,verified=cr?.placement==='image-verified';
 const segs=useMemo(()=>cr?.segments??[],[cr]);
 const loaded=replayImageReady===st.frame_id;
 const frame=catalog?.frames.find(f=>f.id===st.frame_id);
 const calibration=frame?.camera_model?.validation;
 const behind=Math.abs(st.turn_to_route_deg??0)>90;
 return <div className="navcam">
  <div className="navcam-stage"><img key={st.frame_id} src={`/api/frames/${st.frame_id}/display`} alt={`Real Curiosity ${frame?.instrument?.includes('HAZCAM')?'front Hazcam':'Navcam'} frame, sol ${st.sol}`} onLoad={()=>set({replayImageReady:st.frame_id})}/>
   {loaded&&<svg className="camera-route" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Same remaining route as the map, projected with measured camera geometry">{segs.map((seg,i)=><polyline key={i} className={verified?'new':'unverified'} points={seg.map(q=>`${q[0]},${q[1]}`).join(' ')}/>)}</svg>}
   {!loaded&&<div className="replay-loading" role="status"><LoaderCircle className="spin" size={22}/>Loading real camera frame…</div>}
   {loaded&&!segs.length&&<div className="offscreen-guidance"><RotateCcw size={26}/><strong>{behind?'Destination behind the camera':'Route outside this camera view'}</strong><span>{behind?'Follow the return route on the map.':'The map keeps the complete route.'}</span></div>}
  </div>
  <div className="view-caption">Real archived {frame?.instrument?.includes('HAZCAM')?'Front Hazcam':'Navcam'} · {st.time.slice(0,10)} · {calibration?'Calibrated camera + measured terrain':'Measured stereo projection'}<br/>{verified?'Orbital placement image-checked.':'Archive-based map position; independent image match not established.'} Distant route height uses the DEM. Recorded footage, not a live rover.</div>
 </div>;
}

/** Bottom bar in Replay mode: transport controls and the event-marked scrubber. */
export function ReplayTimeline(){
 const s=useMission();
 const [playing,setPlaying]=useState(false);
 const r=s.replay,n=r?.steps.length??0;
 useEffect(()=>{setPlaying(false);},[r?.scenario.id]);
 useEffect(()=>{if(!playing||!r)return;const t=window.setInterval(()=>{const st=useMission.getState();if(st.replayView==='navcam'&&st.replayImageReady!==r.steps[st.replayStep].frame_id)return;if(st.replayStep>=n-1){setPlaying(false);return;}st.set({replayStep:st.replayStep+1,frameId:r.steps[st.replayStep+1].frame_id});},2400);return()=>window.clearInterval(t);},[playing,r,n]);
 if(!r)return <section className="replay-timeline empty"><span>Select a drive replay from the left.</span></section>;
 const go=(i:number)=>s.set({replayStep:Math.max(0,Math.min(n-1,i)),frameId:r.steps[Math.max(0,Math.min(n-1,i))].frame_id});
 return <section className="replay-timeline">
  <div className="transport">
   <button onClick={()=>go(s.replayStep-1)} aria-label="Previous frame"><SkipBack size={16}/></button>
   <button className="play" onClick={()=>{if(s.replayStep>=n-1)go(0);setPlaying(!playing);}} aria-label={playing?'Pause':'Play'}>{playing?<Pause size={18}/>:<Play size={18}/>}</button>
   <button onClick={()=>go(s.replayStep+1)} aria-label="Next frame"><SkipForward size={16}/></button>
  </div>
  <div className="scrubber">
   <input type="range" min={0} max={n-1} value={s.replayStep} onChange={e=>go(Number(e.target.value))} aria-label="Replay position"/>
   <div className="scrubber-marks">{r.events.map(e=><button key={e.step} className={`mark ${e.type}`} style={{left:`${(e.step/(n-1))*100}%`}} onClick={()=>go(e.step)} title={EVENT_TEXT[e.type]}/>)}</div>
   <div className="scrubber-labels"><span>Frame {s.replayStep+1} of {n}</span><span>{r.events.length} re-plans · {r.events.filter(e=>e.type==='u_turn').length} U-turns</span></div>
  </div>
  {s.busy&&<LoaderCircle className="spin" size={16}/>}
 </section>;
}
