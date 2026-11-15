import type {LocalNavigation} from './types';
import {navigationRibbon} from './routeGeometry';

type P=[number,number];
const pts=(ps:P[])=>ps.map(p=>`${p[0].toFixed(5)},${p[1].toFixed(5)}`).join(' ');

export default function RouteOverlay({local,faint,far,destination,visibilityMask}:{local:LocalNavigation|null;faint:boolean;far?:number[][][];destination?:number[]|null;visibilityMask?:string|null}){
 const ribbon=navigationRibbon(local?.ribbons??[],local?.ribbon_s_m,far??[],local?.length_m??0,!faint,local?.status==='u_turn');
 const k=faint?.45:1;
 const top=ribbon.strokes.length?Math.min(...ribbon.strokes.flatMap(s=>s.centre.map(p=>p[1]))):0;
 return <svg className="camera-route route-overlay" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Continuous tapering route; dashed centre beyond the local segment is the map plan, not verified free ground">
  <defs>
   {visibilityMask&&<mask id="camera-visible-ground" maskUnits="userSpaceOnUse" x="0" y="0" width="1" height="1"><image href={visibilityMask} x="0" y="0" width="1" height="1" preserveAspectRatio="none"/></mask>}
   <linearGradient id="route-depth" gradientUnits="userSpaceOnUse" x1="0" y1="1" x2="0" y2={top}>
    <stop offset="0" stopColor="#3ce1ff" stopOpacity={.66*k}/>
    <stop offset=".55" stopColor="#3ce1ff" stopOpacity={.5*k}/>
    <stop offset="1" stopColor="#7cf0ff" stopOpacity={.34*k}/>
   </linearGradient>
   <filter id="route-glow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation=".006"/></filter>
  </defs>
  <g mask={visibilityMask?'url(#camera-visible-ground)':undefined}>
  {(local?.corridor_edges??[]).map((line,i)=><polyline key={`c${i}`} className="corridor-edge" points={line.map(p=>`${p[0]},${p[1]}`).join(' ')}/>)}
  {ribbon.strokes.map((stroke,i)=><g key={i}>
   <polygon className="route-shadow" points={pts(stroke.outline)} filter="url(#route-glow)"/>
   <polygon className="route-paint unified-route" points={pts(stroke.outline)} fill="url(#route-depth)"/>
   {stroke.planned.length>1&&<polyline className="route-planned-centre" points={pts(stroke.planned)}><title>Projected map plan; local clearance not established</title></polyline>}
  </g>)}
  {ribbon.chevrons.map((c,i)=><polyline key={i} className="route-chevron" points={pts(c)} opacity={k}/>)}
  {ribbon.hidden.map((h,i)=><polyline key={`h${i}`} className="route-far-hidden" points={pts(h)} opacity={k}/>)}
  {ribbon.end&&<g className="route-end" opacity={k}><circle cx={ribbon.end[0]} cy={ribbon.end[1]} r=".011" className="route-end-halo"/><circle cx={ribbon.end[0]} cy={ribbon.end[1]} r=".0055" className="route-end-dot"/></g>}
  </g>
  {destination&&<g className="camera-destination" transform={`translate(${destination[0]} ${destination[1]})`} role="img" aria-label="Destination in camera view">
   <title>Destination · calibrated camera projection; map placement uses the archived pose</title>
   <circle r=".006" fill="#f2a65a" stroke="#111b22" strokeWidth=".002"/>
   <path d="M0 0V-.055 M0-.055L.033-.045 .0-.031" fill="#f2a65a" stroke="#fff3df" strokeWidth=".003" strokeLinejoin="round"/>
   <text x={destination[0]>.75?-.012:.041} y="-.032" textAnchor={destination[0]>.75?'end':'start'} fill="#fff3df" stroke="#111b22" strokeWidth=".006" paintOrder="stroke" fontSize=".023" fontWeight="600">Destination</text>
  </g>}
 </svg>;
}

export function TurnCue({direction,degrees}:{direction:'left'|'right';degrees:number}){
 const sweep=direction==='right'?1:0,dx=direction==='right'?1:-1;
 const sx=.5-dx*.14,ex=.5+dx*.14;
 return <svg className="camera-route turn-cue" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label={`Turn in place ${Math.round(degrees)} degrees ${direction}`}>
  <path d={`M ${sx} .9 A .14 .1 0 1 ${sweep} ${ex} .9`} className="turn-arc"/>
  <polygon points={`${ex-dx*.028},.875 ${ex+dx*.012},.9 ${ex-dx*.028},.93`} className="turn-head"/>
 </svg>;
}
