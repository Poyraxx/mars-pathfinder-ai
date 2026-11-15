import {useEffect,useRef,useState} from 'react';
import {Activity,ArrowDownToLine,ArrowUpRight,Crosshair,Eye,Flag,Globe2,Info,Layers3,LoaderCircle,MapPin,Mountain,Orbit,Plus,Radar,Route as RouteIcon,ScanLine,Video,X,ZoomIn} from 'lucide-react';
import {Area,AreaChart,ResponsiveContainer,Tooltip,XAxis,YAxis} from 'recharts';
import CameraView from './CameraView';
import Globe from './Globe';
import {ReplayPanel,ReplayTimeline,ReplayView} from './Replay';
import {EnvironmentPanel,EvaControls,EvaPanel,GeologyAlong,OverlaySwitch,ProfilePicker,ScienceTargets} from './Marswalk';
import {api,useMission} from './store';
import type {Catalog,Event,Mission,OrbitalResult,Point} from './types';

const coord=(p:Point|null)=>p?`${Math.abs(p.lat).toFixed(4)}° ${p.lat<0?'S':'N'} · ${Math.abs(p.lon).toFixed(4)}° ${p.lon<0?'W':'E'}`:'Not set';
const distance=(n:number|undefined)=>n===undefined?'—':n>=1000?`${(n/1000).toFixed(n>=100000?0:1)} km`:`${n.toFixed(0)} m`;
const routeNames:Record<string,string>={distance:'Shortest distance',terrain:'Terrain-aware',surface:'Terrain + camera'};
const famous=['Olympus Mons','Valles Marineris','Jezero','Gale','Hellas Planitia','Elysium Mons'];

function App(){
 const s=useMission();
 const [connected,setConnected]=useState(false),[booting,setBooting]=useState(false),[layerMenu,setLayerMenu]=useState(false),[tab,setTab]=useState<'eva'|'environment'|'profile'|'compare'|'log'>('eva');
 const sourcesRef=useRef<HTMLElement>(null);
 const analysisToggleRef=useRef<HTMLButtonElement>(null);
 useEffect(()=>{
  if(!s.showSources)return;
  const previous=document.activeElement as HTMLElement|null,dialog=sourcesRef.current;
  dialog?.querySelector<HTMLElement>('button')?.focus();
  const onKey=(event:KeyboardEvent)=>{
   if(event.key==='Escape'){s.set({showSources:false});return;}
   if(event.key!=='Tab'||!dialog)return;
   const controls=[...dialog.querySelectorAll<HTMLElement>('button,a[href],input,select,[tabindex="0"]')].filter(el=>el.getClientRects().length>0);
   const first=controls[0],last=controls[controls.length-1];
   if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus();}
   else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus();}
  };
  document.addEventListener('keydown',onKey);
  return()=>{document.removeEventListener('keydown',onKey);previous?.focus();};
 },[s.showSources]);
 async function load(){setBooting(true);try{const catalog=await api<Catalog>('/api/catalog');s.set({catalog,error:null,frameId:s.frameId||catalog.frames[0]?.id||''});setConnected(true);}catch(error){s.set({error:String(error)});setConnected(false);}finally{setBooting(false);}}
 useEffect(()=>{void load();},[]);
 useEffect(()=>{
  if(!s.mission?.id)return;
  const id=s.mission.id,source=new EventSource(`/api/missions/${id}/events`);
  source.onmessage=(message)=>{
   const event=JSON.parse(message.data) as Event,current=useMission.getState();
   if(current.mission?.id!==id)return;
   if(!current.events.some(e=>e.sequence_id===event.sequence_id))s.set({events:[...current.events,event]});
   if(['route.planned','route.replanned','route.unchanged','localization.rejected','job.failed'].includes(event.type))
    void api<Mission>(`/api/missions/${id}/state`).then(m=>{if(useMission.getState().mission?.id===id)s.set({mission:m,busy:false});}).catch(error=>s.set({busy:false,error:String(error)}));
  };
  source.onerror=()=>{if(source.readyState===EventSource.CLOSED)s.set({error:'Event stream disconnected. Recalculate to reconnect.',busy:false});};
  return()=>source.close();
 },[s.mission?.id]);
 async function plan(){
  if(!s.start||!s.goal)return;
  s.set({busy:true,error:null,events:[],mission:null,orbital:null});
  const pr=s.catalog?.profiles?.[s.profileKind];
  const profile=pr?{kind:s.profileKind,speed_m_s:pr.speed_m_s,max_slope_deg:pr.max_slope_deg,clearance_m:pr.clearance_m,obstacle_step_m:pr.obstacle_step_m}:undefined;
  const eva={budget_min:s.budgetMin??pr?.budget_min??420,work_min:s.stops.map((_,i)=>s.stopWork[i]??30),date:s.evaDate};
  try{const created=await api<{id:string;map_version:number}>('/api/missions',{region_id:s.region?s.region.id:'global',start:s.start,goal:s.goal,stops:s.stops,avoid_features:s.avoid,...(profile?{profile}:{}),eva});
   s.set({mission:await api<Mission>(`/api/missions/${created.id}/state`)});
   await api(`/api/missions/${created.id}/plan`,{expected_map_version:created.map_version});
  }catch(error){s.set({busy:false,error:error instanceof Error?error.message:String(error)});}
 }
 async function analyze(){if(!s.mission||!s.frameId)return;s.set({busy:true,error:null});try{await api(`/api/missions/${s.mission.id}/observations`,{frame_id:s.frameId,base_map_version:s.mission.map_version});}catch(error){s.set({busy:false,error:error instanceof Error?error.message:String(error)});}}
 async function orbitalAnalysis(){const m=s.mission;if(!m)return;s.set({busy:true,error:null});try{const r=await api<OrbitalResult>(`/api/missions/${m.id}/orbital-analysis?mode=${s.activeRoute}`,{});s.set({orbital:{...r,mission_id:m.id},busy:false});}catch(error){s.set({busy:false,error:error instanceof Error?error.message:String(error)});}}
 async function download(){if(!s.mission)return;try{const report=await api(`/api/missions/${s.mission.id}/report`);const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`mars-mission-${s.mission.id.slice(0,8)}.json`;a.click();URL.revokeObjectURL(url);}catch(error){s.set({error:String(error)});}}
 const flyTo=(name:string)=>{const f=s.features.find(x=>x.name===name);if(f)s.set({feature:f,exploreTarget:{lat:f.lat,lon:f.lon,alt:Math.min(9000000,Math.max(60000,f.diameter_km*2600))},region:null,view:'orbital'});};
 const setMode=(mode:'explore'|'plan'|'replay')=>{
  const overview=mode==='explore'||(mode==='plan'&&s.mode==='replay'&&!s.region);
  s.set({mode,pick:null,replayGoalPick:false,view:mode==='replay'||overview?'orbital':s.view,
   ...(mode==='replay'||mode==='explore'?{region:null}:{}),
   ...(overview?{feature:null,exploreTarget:{lat:17,lon:42,alt:10000000}}:{})});
 };
 const openGale=()=>{const r=s.catalog?.regions[0];if(r){s.selectRegion(r);s.set({mode:'plan'});}};
 const active=s.mission?.routes.find(r=>r.mode===s.activeRoute&&r.status==='ready');
 const frame=s.catalog?.frames.find(f=>f.id===s.frameId);
 const observation=s.mission?[...s.mission.observations].reverse().find(o=>o.frame_id===s.frameId):undefined;
 const busy=s.busy||s.mission?.state==='planning'||s.mission?.state==='perceiving';
 const cameraView=s.mode==='plan'&&!!s.region&&(s.view==='astronaut'||s.view==='dual');
 const mm=s.catalog?.model.metrics as {miou?:number;recall?:Record<string,number>;precision?:Record<string,number>}|null|undefined;
 const showDetails=s.mode==='plan'&&s.showAI;
 return <div className={`app mode-${s.mode} ${showDetails?'has-details':''} ${s.mode!=='explore'?'has-bottom':''}`}>
  <a className="skip-link" href="#mission-workspace">Skip to workspace</a>
  <header className="topbar">
   <a className="brand" href="/" aria-label="Mars Pathfinder home"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 21 20H3Z" fill="none" stroke="currentColor" strokeWidth="1.6"/><circle cx="12" cy="14" r="2.2" fill="currentColor"/></svg><span>Pathfinder</span></a>
   <nav className="modes" aria-label="Mode">{([['plan','Plan a route',RouteIcon],['replay','Live GPS Test',Radar],['explore','Explore',Globe2]] as const).map(([id,label,Icon])=><button key={id} className={s.mode===id?'active':''} aria-current={s.mode===id?'page':undefined} onClick={()=>setMode(id)}><Icon size={15}/>{label}</button>)}</nav>
   <div className="topbar-right">
    <span className={`connection ${connected?'online':''}`}><i/>{connected?'Server live':booting?'Waking server…':'Offline'}</span>
    <button className="text-button" onClick={()=>s.set({showSources:true})}><Info size={15}/>Sources & limits</button>
   </div>
  </header>

  <aside className="sidebar" aria-label={s.mode==='plan'?'Route setup':s.mode==='replay'?'Test drive selection':'Mars places'}>
   {s.mode==='explore'&&<div className="panel-section">
    <h2 className="panel-title">Explore Mars</h2>
    <label className="search"><MapPin size={15}/><input list="mars-features" placeholder={`Search ${s.features.length||''} named places`} aria-label="Search named Mars features" onChange={e=>{const f=s.features.find(x=>x.name.toLowerCase()===e.target.value.toLowerCase());if(f)flyTo(f.name);}}/></label>
    <datalist id="mars-features">{s.features.map((f,i)=><option key={i} value={f.name}>{f.type} · {f.diameter_km.toFixed(0)} km</option>)}</datalist>
    {s.feature&&<div className="feature-card"><strong>{s.feature.name}</strong><span>{s.feature.type} · {s.feature.diameter_km.toLocaleString('en-GB',{maximumFractionDigits:1})} km across</span><small>{coord(s.feature)} · IAU name</small>
     <div className="button-row"><button className="ghost-button" onClick={()=>s.set({mode:'plan',planMode:'global',region:null,start:null,goal:null,mission:null,pick:'start'})}><RouteIcon size={14}/>Plan a route here</button>{!s.avoid.includes(s.feature.name)&&<button className="ghost-button" onClick={()=>s.set({avoid:[...s.avoid,s.feature!.name].slice(0,10)})}>Avoid on routes</button>}</div>
     <a href={s.feature.link} target="_blank" rel="noreferrer">Gazetteer entry<ArrowUpRight size={12}/></a></div>}
    <ul className="place-list">{famous.map(name=>{const f=s.features.find(x=>x.name===name);return <li key={name}><button disabled={!f} onClick={()=>flyTo(name)}><span>{name}</span><small>{f?`${f.type} · ${f.diameter_km.toFixed(0)} km`:''}</small></button></li>;})}</ul>
    <button className="feature-cta" onClick={openGale} disabled={!s.catalog?.regions.length}><span className="thumb"/><span><strong>Gale Crater · Rocknest</strong><small>1 m terrain, rover cameras, live hazard fusion</small></span><ArrowUpRight size={16}/></button>
    <button className="feature-cta" onClick={()=>setMode('replay')}><span className="thumb replay"/><span><strong>Live GPS Test</strong><small>AI route guidance on real NASA rover images</small></span><ArrowUpRight size={16}/></button>
   </div>}

   {s.mode==='plan'&&<div className="panel-section">
    <h2 className="panel-title">Plan a route</h2>
    <p className="panel-lede">{s.region?'Gale · Rocknest: 1 m HiRISE terrain with rover cameras.':'Anywhere on Mars: live MOLA/HRSC 200 m terrain. Regional context only, no rock-level hazards.'}</p>
    <div className="segmented" role="group" aria-label="Terrain data"><button className={!s.region?'active':''} onClick={()=>s.set({region:null,planMode:'global',start:null,goal:null,stops:[],mission:null,view:'orbital'})}>Anywhere</button><button className={s.region?'active':''} onClick={openGale}>Gale 1 m</button></div>
    <div className="waypoints">
     <button className={`waypoint ${s.pick==='start'?'picking':''}`} disabled={busy} onClick={()=>s.set({pick:s.pick==='start'?null:'start'})}><i className="dot start"/><span><strong>{s.pick==='start'?'Click the map…':'Start'}</strong><small>{coord(s.start)}</small></span><Crosshair size={15}/></button>
     {s.stops.map((p,i)=><div className="waypoint stop" key={i}><i className="dot stop"/><span><strong>{s.stopNames[i]??`Stop ${i+1}`}</strong><small>{coord(p)}</small></span><button aria-label={`Remove stop ${i+1}`} onClick={()=>s.set({stops:s.stops.filter((_,j)=>i!==j),stopWork:s.stopWork.filter((_,j)=>i!==j),stopNames:s.stopNames.filter((_,j)=>i!==j),mission:null})}><X size={14}/></button></div>)}
     <button className={`waypoint ${s.pick==='goal'?'picking':''}`} disabled={busy} onClick={()=>s.set({pick:s.pick==='goal'?null:'goal'})}><i className="dot goal"/><span><strong>{s.pick==='goal'?'Click the map…':'Destination'}</strong><small>{coord(s.goal)}</small></span><Flag size={15}/></button>
    </div>
    <button className="link-button" disabled={s.stops.length>=5||busy} onClick={()=>s.set({pick:'stop'})}><Plus size={14}/>Add a stop on the map ({s.stops.length}/5)</button>
    <ScienceTargets/>
    <ProfilePicker/>
    <EvaControls/>
    {s.avoid.length>0&&<div className="chips"><span>Avoiding</span>{s.avoid.map(n=><button key={n} className="chip" onClick={()=>s.set({avoid:s.avoid.filter(x=>x!==n),mission:null})}>{n}<X size={11}/></button>)}</div>}
    <button className="primary-button" disabled={!s.start||!s.goal||busy||!connected} onClick={plan}>{busy?<LoaderCircle className="spin" size={16}/>:<RouteIcon size={16}/>}{busy?'Calculating…':'Calculate route'}</button>
    {!active&&<p className="setup-hint"><MapPin size={16}/><span>{s.pick?'Select a location on the map. Drag to rotate; scroll to zoom.':!s.start?'Choose Start, then select a point on the map.':!s.goal?'Choose Destination, then select a second point.':'Your points are set. Calculate to compare routes.'}</span></p>}
    {active&&<div className="route-summary">
     <div className="metric-row"><div><small>Distance</small><strong>{distance(active.distance_m)}</strong></div><div><small>Climb</small><strong>{distance(active.ascent_m)}</strong></div><div><small>Max slope</small><strong>{active.max_slope_deg?.toFixed(1)}°</strong></div></div>
     <select aria-label="Route type" value={s.activeRoute} onChange={e=>s.set({activeRoute:e.target.value})}>{Object.entries(routeNames).map(([k,v])=><option key={k} value={k}>{v}</option>)}</select>
    </div>}
    {active?.features&&active.features.length>0&&<div className="route-features"><h3>Named places on this route</h3>{active.features.map(f=><div key={f.name} className={`route-feature ${f.relation==='crosses'?'crosses':''}`}><span><strong>{f.name}</strong><small>{f.relation==='crosses'?'Crosses':'Passes near'} · {f.type} · {f.diameter_km.toFixed(0)} km · at {f.along_km.toFixed(1)} km</small></span>{!s.avoid.includes(f.name)&&<button className="ghost-button small" onClick={()=>s.set({avoid:[...s.avoid,f.name].slice(0,10),mission:null})}>Avoid</button>}</div>)}</div>}
    <GeologyAlong/>
   </div>}

   {s.mode==='replay'&&<ReplayPanel/>}
  </aside>

  <main id="mission-workspace" tabIndex={-1} className={`stage ${s.mode==='replay'?'split':''}`}>
   <>
   {s.mode==='replay'&&<ReplayView/>}
   <div className="map-area">
    <Globe/>
    {s.mode==='plan'&&s.region&&<div className="view-switcher" role="group" aria-label="View">{([['orbital','Orbit',Globe2],['terrain','Terrain',Mountain],['astronaut','Camera',Video],['dual','Camera + map',ScanLine]] as const).map(([id,label,Icon])=><button key={id} className={s.view===id?'active':''} onClick={()=>s.set({view:id})}><Icon size={14}/>{label}</button>)}</div>}
    {s.mode==='explore'&&!s.feature&&<div className="intro"><h1>Explore Mars, plan a route, test rover guidance.</h1><p>Real NASA and USGS data: 200 m global terrain, 2055 named places, and Curiosity's own camera frames.</p></div>}
    {s.mode==='replay'&&<span className="map-label">Map · real rover track</span>}
    <OverlaySwitch/>
    {s.mode!=='replay'&&<div className="relief-switch" role="group" aria-label="Map style"><button className={!s.relief?'active':''} aria-pressed={!s.relief} onClick={()=>s.set({relief:false,layer:'viking'})}>Imagery</button><button className={s.relief?'active':''} aria-pressed={s.relief} onClick={()=>s.set({relief:true,layer:'mola'})}><Mountain size={14}/>Relief</button></div>}
    {s.relief&&s.mode!=='replay'&&<div className="relief-legend"><strong>Elevation · MOLA/HRSC</strong><div className="relief-bar"/><div className="relief-ticks"><span>−8 km</span><span>0</span><span>+10 km</span><span>+21 km</span></div><small>Heights exaggerated ×12 so mountains rise and craters sink. Colour = MOLA elevation; light and shadow computed from the same DEM (sun from the north-west).</small>{s.mission?.routes.some(r=>r.status==='ready')&&<p className="relief-route"><i/>Route: lowest slope + roughness cost on the same 200 m MOLA/HRSC terrain. Steep slopes cost more, so it picks gentler ways down and up.</p>}</div>}
    <div className="map-tools">
     <div className="layer-control"><button className={layerMenu?'active':''} onClick={()=>setLayerMenu(!layerMenu)} aria-label="Map layers" aria-expanded={layerMenu}><Layers3 size={17}/></button>{layerMenu&&<div className="layer-popover">{s.catalog?.layers.map(l=><button key={l.id} onClick={()=>s.set({layer:l.id})} className={s.layer===l.id?'selected':''}><strong>{l.name}</strong><small>{l.kind} · {l.resolution_m} m/px</small></button>)}<label className="check"><input type="checkbox" checked={s.showRisk} disabled={!s.region} onChange={e=>s.set({showRisk:e.target.checked})}/>Slope overlay (Gale)</label><p>Thermal mosaic brightness is not current temperature.</p></div>}</div>
     <button onClick={()=>s.set({view:'orbital',exploreTarget:{lat:17,lon:42,alt:10000000},feature:null})} aria-label="Whole planet"><Globe2 size={17}/></button>
     {s.mode==='replay'&&<button onClick={()=>{s.set({followRover:false});window.dispatchEvent(new Event('replay-fit'));}} aria-label="Fit rover and route" title="Fit whole drive"><Crosshair size={17}/></button>}
     {s.mode==='replay'&&<button className={s.followRover?'active':''} aria-pressed={s.followRover} onClick={()=>{const on=!s.followRover;s.set({followRover:on});if(on)window.dispatchEvent(new Event('replay-rover'));}} aria-label="Zoom to rover and follow it" title="Zoom to rover and follow it"><ZoomIn size={17}/></button>}
     {s.mode==='plan'&&<button ref={analysisToggleRef} className={s.showAI?'active':''} onClick={()=>s.set({showAI:!s.showAI})} aria-label="Route analysis panel" aria-controls="route-analysis" aria-expanded={s.showAI}><Radar size={17}/></button>}
    </div>
    {cameraView&&<CameraView dual={s.view==='dual'}/>}
   </div></>
   {s.error&&<div className="error-banner" role="alert"><Info size={16}/><span>{s.error}</span><button onClick={()=>s.set({error:null})} aria-label="Dismiss error"><X size={15}/></button></div>}
   {!connected&&<div className="service-overlay"><Radar size={26}/><h2>{booting?'Waking the server':'Server unavailable'}</h2><p>The free server sleeps when idle and needs about a minute to wake.</p><button className="primary-button" onClick={load} disabled={booting}>{booting?<LoaderCircle className="spin" size={15}/>:<Activity size={15}/>}Try again</button></div>}
  </main>

  {showDetails&&<aside id="route-analysis" className="details" aria-label="Route analysis">
   <div className="details-header"><strong>Route analysis</strong><button aria-label="Close route analysis" onClick={()=>{s.set({showAI:false});analysisToggleRef.current?.focus();}}><X size={16}/></button></div>
   <section><h3><Orbit size={15}/>From orbit</h3>
    <dl><div><dt>Terrain</dt><dd>{s.region?'HiRISE 1 m DTM':s.mission?.region?`MOLA/HRSC · ${Math.round(s.mission.region.cell_m)} m grid`:'MOLA/HRSC 200 m'}</dd></div><div><dt>Route cost</dt><dd>Slope + roughness (deterministic)</dd></div><div><dt>Landmark model</dt><dd>{s.catalog?.orbital_model?.ready?`HiRISE · test F1 ${(s.catalog.orbital_model.test!.macro_f1*100).toFixed(0)}%`:'Not installed'}</dd></div></dl>
    {s.catalog?.orbital_model?.ready&&<button className="ghost-button" disabled={!active||busy} onClick={orbitalAnalysis}><Orbit size={14}/>Classify the route from orbit</button>}
    {s.orbital&&s.orbital.mission_id===s.mission?.id&&<div className="orbital-result"><small>{s.orbital.windows.length} CTX windows · {s.orbital.window_m.toFixed(0)} m · {s.orbital.elapsed_ms.toFixed(0)} ms</small><div className="chips">{Object.entries(s.orbital.class_counts).filter(([,n])=>n>0).sort((a,b)=>b[1]-a[1]).map(([k,n])=><span key={k} className={`chip c-${k}`}>{k.replaceAll('_',' ')} {n}</span>)}</div><p className="fine-print">Landmark context only; trained on HiRISE, applied to CTX. Does not change route costs.</p></div>}
   </section>
   {s.region&&<section><h3><ScanLine size={15}/>From the rover</h3>
    {s.catalog?.frames.length?<select aria-label="Camera frame" value={s.frameId} onChange={e=>s.set({frameId:e.target.value})}>{(['drive','panorama','approach'] as const).map(seq=>{const list=s.catalog!.frames.filter(f=>(f.sequence??'drive')===seq).sort((a,b)=>(a.sequence_index??0)-(b.sequence_index??0));return list.length?<optgroup key={seq} label={seq==='drive'?'Sol 57 drive':seq==='panorama'?'Sol 57 panorama':'Sols 55–56 approach'}>{list.map(f=><option value={f.id} key={f.id}>{seq==='panorama'?'Pan':'Frame'} {String(f.sequence_index??0).padStart(2,'0')}{f.registration.verified?' · verified':''}</option>)}</optgroup>:null;})}</select>:null}
    <dl><div><dt>Placement</dt><dd className={frame?.registration.verified?'ok':'warn'}>{frame?.registration.verified?`Image-matched ±${frame.registration.horizontal_error_bound_m?.toFixed(2)} m`:'Not independently verified'}</dd></div><div><dt>Obstacles</dt><dd>{observation?.localization.hazard_pixels!==undefined?`${observation.localization.hazard_pixels} px ≥ ${observation.localization.step_threshold_m?.toFixed(2)} m`:'Stereo height steps'}</dd></div><div><dt>Inference</dt><dd>{observation?`${observation.inference_ms.toFixed(0)} ms`:'Not run'}</dd></div></dl>
    <button className="ghost-button" onClick={()=>{void analyze();if(s.view==='orbital'||s.view==='terrain')s.set({view:'dual'});}} disabled={!s.mission||!frame||busy||!!observation}><Eye size={14}/>{observation?'Frame processed':'Analyse this frame'}</button>
    <p className="fine-print">{mm?.miou!==undefined?`Segmentation test: ${(mm.miou*100).toFixed(1)}% mIoU, big-rock recall ${((mm.recall?.big_rock??0)*100).toFixed(0)}%. `:''}The model is context only; the map uses measured stereo height steps.</p>
   </section>}
  </aside>}

  {s.mode==='replay'&&<ReplayTimeline/>}
  {s.mode==='plan'&&<section className="analysis">
   <div className="tabs" role="tablist">{([['eva','Round trip'],['environment','Conditions'],['profile','Elevation'],['compare','Compare routes'],['log',`Decisions (${s.events.length})`]] as const).map(([id,label])=><button key={id} role="tab" aria-selected={tab===id} className={tab===id?'active':''} onClick={()=>setTab(id)}>{label}</button>)}<button className="text-button" onClick={download} disabled={!s.mission}><ArrowDownToLine size={14}/>Report</button></div>
   <div className="analysis-body">{tab==='eva'?<EvaPanel/>:tab==='environment'?<EnvironmentPanel/>:tab==='profile'?(active?.profile?<ResponsiveContainer width="100%" height="100%"><AreaChart data={active.profile} margin={{top:8,right:16,bottom:0,left:0}}><defs><linearGradient id="fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#5fd4c4" stopOpacity={.28}/><stop offset="100%" stopColor="#5fd4c4" stopOpacity={0}/></linearGradient></defs><XAxis dataKey="distance_m" type="number" tick={{fontSize:11,fill:'#8796a0'}} tickFormatter={v=>distance(v)} axisLine={false} tickLine={false}/><YAxis domain={['dataMin-1','dataMax+1']} width={56} tick={{fontSize:11,fill:'#8796a0'}} tickFormatter={v=>`${Number(v).toFixed(0)} m`} axisLine={false} tickLine={false}/><Tooltip contentStyle={{background:'#111a21',border:'1px solid #26343d',borderRadius:8,fontSize:12}} labelFormatter={v=>`${distance(Number(v))} along route`} formatter={v=>[`${Number(v).toFixed(1)} m`,'Elevation']}/><Area type="linear" dataKey="elevation_m" stroke="#5fd4c4" fill="url(#fill)" strokeWidth={2} isAnimationActive={false}/></AreaChart></ResponsiveContainer>:<p className="empty">Set a start and destination, then calculate to see the elevation profile.</p>)
    :tab==='compare'?<table className="compare"><thead><tr><th>Route</th><th>Distance</th><th>Climb</th><th>Max slope</th><th>Compute</th></tr></thead><tbody>{s.mission?.routes.map(r=><tr key={r.mode} className={r.mode===s.activeRoute?'selected':''} onClick={()=>s.set({activeRoute:r.mode})}><td>{routeNames[r.mode]}</td><td>{r.status==='ready'?distance(r.distance_m):r.status.replaceAll('_',' ')}</td><td>{distance(r.ascent_m)}</td><td>{r.max_slope_deg!==undefined?`${r.max_slope_deg}°`:'—'}</td><td>{r.planning_ms!==undefined?`${r.planning_ms} ms`:'—'}</td></tr>)}</tbody></table>
    :<ol className="log">{s.events.map(e=><li key={e.sequence_id}><time>{new Date(e.timestamp).toLocaleTimeString('en-GB')}</time><span>{e.message}</span>{e.map_version!==undefined&&<small>map v{e.map_version}</small>}</li>)}{!s.events.length&&<p className="empty">Decisions appear here as the server computes them.</p>}</ol>}</div>
  </section>}

  {s.showSources&&<div className="modal-backdrop" onClick={()=>s.set({showSources:false})}><section ref={sourcesRef} className="sources-modal" role="dialog" aria-modal="true" aria-label="Sources and limits" onClick={e=>e.stopPropagation()}>
   <button className="modal-close" aria-label="Close" onClick={()=>s.set({showSources:false})}><X size={18}/></button>
   <h2>Sources and limits</h2>
   <p>Every number in this app is computed on the server from archived NASA, USGS and ESA data. Rover images are historical, not a live feed.</p>
   <div className="source-list">{s.catalog?.sources.map(source=><a href={source.url} key={source.name} target="_blank" rel="noreferrer"><strong>{source.name}</strong><span>{source.usage}</span><ArrowUpRight size={14}/></a>)}</div>
   <h3>What this prototype does not establish</h3>
   <ul><li>Human EVA safety, suit consumables, radiation or medical fitness.</li><li>Rock-level hazards outside the Gale 1 m area; the global terrain is 200 m and coarser on long routes.</li><li>Camera placement for frames that did not pass the image-correlation registration test; those never update the map.</li><li>Reliable big-rock detection by the learned model; map obstacles come from stereo height steps and can miss low or distant rocks.</li><li>Real outlines of named places: avoidance uses a circle of the published diameter.</li></ul>
   <p className="fine-print">Original implementation assisted by OpenAI Codex. No NASA endorsement is implied.</p>
  </section></div>}
 </div>;
}
export default App;
