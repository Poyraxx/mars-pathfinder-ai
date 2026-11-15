import {useEffect,useState} from 'react';
import {ArrowUpRight,Bike,Footprints,Layers,Plus,Truck} from 'lucide-react';
import {api,useMission} from './store';
import type {EnvironmentReport,ScienceTarget} from './types';

const minutes=(m:number|undefined|null)=>m==null?'—':Math.abs(m)>=1440?`${(m/1440).toFixed(1)} d`:Math.abs(m)>=90?`${(m/60).toFixed(1)} h`:`${Math.round(m)} min`;
const PROFILE_ICONS={astronaut:Footprints,crew_cart:Truck,rover:Bike} as const;
const SHORT={astronaut:'Astronaut',crew_cart:'Crew + cart',rover:'Rover'} as const;

/** Movement profile: who or what travels. Assumptions are shown, not hidden. */
export function ProfilePicker(){
 const s=useMission();
 const profiles=s.catalog?.profiles;if(!profiles)return null;
 const p=profiles[s.profileKind];
 return <div className="marswalk-block">
  <h3>Who travels</h3>
  <div className="segmented three" role="group" aria-label="Movement profile">{(Object.keys(profiles) as (keyof typeof PROFILE_ICONS)[]).map(k=>{const Icon=PROFILE_ICONS[k];return <button key={k} className={s.profileKind===k?'active':''} onClick={()=>s.set({profileKind:k,mission:null,budgetMin:null})}><Icon size={14}/>{SHORT[k]}</button>;})}</div>
  <dl className="profile-facts"><div><dt>Speed</dt><dd>{p.speed_m_s} m/s</dd></div><div><dt>Max slope</dt><dd>{p.max_slope_deg}°</dd></div><div><dt>Clearance</dt><dd>{p.clearance_m} m</dd></div><div><dt>Step</dt><dd>{p.obstacle_step_m} m</dd></div></dl>
  <p className="fine-print">{p.basis}</p>
 </div>;
}

/** Sourced science stops for the current area, added to the route with their work time. */
export function ScienceTargets(){
 const s=useMission();
 useEffect(()=>{if(!s.scienceTargets)void api<{gale:ScienceTarget[];global:ScienceTarget[]}>('/api/science-targets').then(t=>s.set({scienceTargets:t})).catch(()=>{});},[s.scienceTargets]);
 const list=(s.region?s.scienceTargets?.gale:s.scienceTargets?.global)??[];
 if(!list.length)return null;
 const add=(t:ScienceTarget)=>{if(s.stops.length>=5)return;s.set({stops:[...s.stops,{lat:t.lat,lon:t.lon}],stopWork:[...s.stopWork.slice(0,s.stops.length),t.work_min],stopNames:[...s.stopNames.slice(0,s.stops.length),t.name],mission:null});};
 return <details className="marswalk-block science-details" open={s.stops.length===0&&!!s.region}>
  <summary><h3>Science stops · {list.length} sourced {s.region?'Curiosity targets':'landing sites'}</h3></summary>
  <ul className="science-list">{list.map(t=>{const added=s.stopNames.includes(t.name);return <li key={t.id}>
   <div className="science-head"><strong>{t.name}</strong>{!added&&<button className="ghost-button small" disabled={s.stops.length>=5} onClick={()=>add(t)}><Plus size={12}/>Add · {t.work_min} min</button>}{added&&<span className="tag">On route</span>}</div>
   <p><b>Why:</b> {t.why}</p><p><b>Do:</b> {t.measure}</p>
   <small>{t.position_source} · {t.sources.map((src,i)=><a key={i} href={src.url} target="_blank" rel="noreferrer">{src.label}<ArrowUpRight size={10}/></a>)}</small>
  </li>;})}</ul>
 </details>;
}

/** Time budget, date and per-stop work minutes that feed the round-trip plan. */
export function EvaControls(){
 const s=useMission();
 const profile=s.catalog?.profiles?.[s.profileKind];
 return <div className="marswalk-block">
  <h3>Round trip</h3>
  <div className="eva-inputs">
   <label>Time budget<span><input type="number" min={10} max={10080} step={10} value={s.budgetMin??profile?.budget_min??420} onChange={e=>s.set({budgetMin:Number(e.target.value)||null,mission:null})}/>min</span></label>
   <label>Date (Earth UTC)<input type="date" value={s.evaDate} onChange={e=>s.set({evaDate:e.target.value})}/></label>
  </div>
  {s.stops.length>0&&<div className="stop-work">{s.stops.map((_,i)=><label key={i}>{s.stopNames[i]??`Stop ${i+1}`}<span><input type="number" min={0} max={600} step={5} value={s.stopWork[i]??30} onChange={e=>{const w=[...s.stopWork];w[i]=Number(e.target.value)||0;s.set({stopWork:w,mission:null});}}/>min work</span></label>)}</div>}
  <p className="fine-print">Return goes back to the start. Budget is a planning figure, not a suit consumables model.</p>
 </div>;
}

/** Analysis tab: outbound, science work and return against the budget, and what each stop costs. */
export function EvaPanel(){
 const e=useMission(st=>st.mission?.eva);
 const names=useMission(st=>st.stopNames);
 if(!e)return <p className="empty">Calculate a route to see the round trip, work time and reserve.</p>;
 if(e.status!=='ready')return <p className="empty">Round trip could not be planned ({e.status.replaceAll('_',' ')}).</p>;
 const total=e.budget_min!,out=e.outbound!.time_min,work=e.stops!.reduce((a,b)=>a+b.work_min,0),back=e.return!.time_min,reserve=e.reserve_min!;
 const scale=Math.max(total,out+work+back);
 const pct=(m:number)=>`${Math.max(0,m)/scale*100}%`;
 const over=reserve<0,low=!over&&(e.reserve_fraction??1)<.2;
 return <div className="eva-panel">
  <div className="eva-head"><strong>{e.profile}</strong><span className={over?'bad':low?'warn':'ok'}>{over?`Over budget by ${minutes(-reserve)}`:`Reserve ${minutes(reserve)} (${Math.round((e.reserve_fraction??0)*100)}%)`}</span></div>
  <div className="eva-bar" role="img" aria-label={`Outbound ${minutes(out)}, work ${minutes(work)}, return ${minutes(back)}, budget ${minutes(total)}`}>
   <i className="out" style={{width:pct(out)}}/><i className="work" style={{width:pct(work)}}/><i className="back" style={{width:pct(back)}}/>{!over&&<i className="reserve" style={{width:pct(reserve)}}/>}
   <b className="budget-mark" style={{left:pct(total)}}/>
  </div>
  <div className="eva-legend"><span><i className="out"/>Out {distanceKm(e.outbound!.distance_m)} · {minutes(out)}</span><span><i className="work"/>Science work {minutes(work)}</span><span><i className="back"/>Return {distanceKm(e.return!.distance_m)} · {minutes(back)}</span><span><i className="reserve"/>Budget {minutes(total)}</span></div>
  {e.stop_marginal&&e.stop_marginal.length>0&&<table className="compare"><thead><tr><th>Stop</th><th>Detour</th><th>Costs (walk + work)</th><th>Reserve without it</th></tr></thead><tbody>{e.stop_marginal.map(m=><tr key={m.stop}><td>{names[m.stop-1]??`Stop ${m.stop}`}</td><td>{m.extra_distance_m==null?'—':distanceKm(m.extra_distance_m)}</td><td>{minutes(m.extra_time_min)}</td><td>{minutes(m.reserve_without_min)}</td></tr>)}</tbody></table>}
  <p className="fine-print">{e.basis} {e.note} Dashed orange line on the map: return leg.</p>
 </div>;
}
const distanceKm=(m:number)=>m>=1000?`${(m/1000).toFixed(1)} km`:`${Math.round(m)} m`;

const KIND_LABEL={measurement:'Measurement',historical:'Historical record',model:'Model'} as const;
/** Analysis tab: sourced environment at the route midpoint, each value labelled by kind. */
export function EnvironmentPanel(){
 const s=useMission();
 const route=s.mission?.routes.find(r=>r.mode===s.activeRoute&&r.status==='ready');
 const mid=route?.points[Math.floor(route.points.length/2)]??s.goal??s.start;
 const elev=route?.profile?.[Math.floor((route.profile.length-1)/2)]?.elevation_m;
 const [report,setReport]=useState<EnvironmentReport|null>(null);
 useEffect(()=>{if(!mid){setReport(null);return;}const q=new URLSearchParams({lat:String(mid.lat),lon:String(mid.lon),date:s.evaDate,...(elev!==undefined?{elevation_m:String(elev)}:{})});void api<EnvironmentReport>(`/api/environment?${q}`).then(setReport).catch(()=>setReport(null));},[mid?.lat,mid?.lon,elev,s.evaDate]);
 if(!mid)return <p className="empty">Set a start point to see conditions.</p>;
 if(!report)return <p className="empty">Loading conditions…</p>;
 return <div className="env-panel">
  <ul>{report.items.map(i=><li key={i.key}><span className={`kind ${i.kind}`}>{KIND_LABEL[i.kind]}</span><div><small>{i.label}</small><strong>{i.value}</strong><a href={i.url} target="_blank" rel="noreferrer">{i.source}<ArrowUpRight size={10}/></a></div></li>)}</ul>
  <p className="fine-print">{route?'Route midpoint':'Selected point'} · {report.date_utc.slice(0,10)} · {report.note}</p>
 </div>;
}

/** Geologic units crossed by the active route (USGS SIM 3292). */
export function GeologyAlong(){
 const s=useMission();
 const g=s.mission?.routes.find(r=>r.mode===s.activeRoute&&r.status==='ready')?.geology;
 if(!g?.segments.length)return null;
 return <div className="route-features geology-along"><h3>Geology on this route</h3>
  {g.segments.map((u,i)=><div key={i} className="route-feature"><i className="swatch" style={{background:u.colour}}/><span><strong>{u.unit} · {u.description}</strong><small>{u.period} · {u.from_km.toFixed(1)}–{u.to_km.toFixed(1)} km</small></span></div>)}
  <p className="fine-print"><a href={g.url} target="_blank" rel="noreferrer">{g.source}<ArrowUpRight size={10}/></a>. {g.note}</p>
 </div>;
}

/** Map overlay switch (Google Earth style): one conditions layer at a time, with its legend and source. */
export function OverlaySwitch(){
 const s=useMission();
 const overlays=s.catalog?.overlays??[];
 const [open,setOpen]=useState(false);
 const active=overlays.find(o=>o.id===s.overlay);
 if(!overlays.length||s.mode==='replay')return null;
 return <>
  <div className="overlay-switch" role="group" aria-label="Conditions layers">
   <button className={open||active?'active':''} aria-expanded={open} onClick={()=>setOpen(!open)}><Layers size={14}/>{active?active.name:'Conditions'}</button>
   {open&&<div className="overlay-menu">{overlays.map(o=><button key={o.id} aria-pressed={s.overlay===o.id} className={s.overlay===o.id?'active':''} onClick={()=>{s.set({overlay:s.overlay===o.id?null:o.id});setOpen(false);}}>{o.name}<small>{o.kind==='measurement'?'Orbital measurement':o.kind==='map'?'Map':o.kind}{o.resolution_m?` · ${o.resolution_m>=1000?`${(o.resolution_m/1000).toFixed(1)} km`:`${o.resolution_m} m`}`:''}</small></button>)}{s.overlay&&<button className="clear" onClick={()=>{s.set({overlay:null});setOpen(false);}}>Hide layer</button>}</div>}
  </div>
  {active&&<div className="overlay-legend"><strong>{active.name}</strong><small>{active.legend}</small><a href={active.url} target="_blank" rel="noreferrer">{active.source}<ArrowUpRight size={10}/></a></div>}
 </>;
}
