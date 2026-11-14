import type {LocalNavigation} from './types';
import {groundArrow} from './routeGeometry';

const pts=(ps:[number,number][])=>ps.map(p=>`${p[0].toFixed(4)},${p[1].toFixed(4)}`).join(' ');

/** Route painted on the ground: one continuous perspective ribbon from the rover, with surface chevrons. */
export default function RouteOverlay({local,faint}:{local:LocalNavigation;faint:boolean}){
 const arrow=groundArrow(local.ribbons,local.ribbon_s_m);
 const k=faint?.45:1;
 const top=arrow?Math.min(...arrow.outline.map(p=>p[1])):0;
 return <svg className="camera-route route-overlay" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Rover route painted on the ground: from the rover, along the collision-checked corridor">
  <defs>
   <linearGradient id="route-depth" gradientUnits="userSpaceOnUse" x1="0" y1="1" x2="0" y2={top}>
    <stop offset="0" stopColor="#3ce1ff" stopOpacity={.66*k}/>
    <stop offset=".55" stopColor="#3ce1ff" stopOpacity={.5*k}/>
    <stop offset="1" stopColor="#7cf0ff" stopOpacity={.34*k}/>
   </linearGradient>
   <filter id="route-glow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation=".006"/></filter>
  </defs>
  {(local.corridor_edges??[]).map((line,i)=><polyline key={`c${i}`} className="corridor-edge" points={line.map(p=>`${p[0]},${p[1]}`).join(' ')}/>)}
  {arrow&&<>
   <polygon className="route-shadow" points={pts(arrow.outline)} filter="url(#route-glow)"/>
   <polygon className="route-paint" points={pts(arrow.outline)} fill="url(#route-depth)"/>
   {arrow.chevrons.map((c,i)=><polyline key={i} className="route-chevron" points={pts(c)} opacity={k}/>)}
  </>}
 </svg>;
}

/** In-place rotation cue when the destination is behind the camera's view. A guidance icon, not a ground route. */
export function TurnCue({direction,degrees}:{direction:'left'|'right';degrees:number}){
 const sweep=direction==='right'?1:0,dx=direction==='right'?1:-1;
 const sx=.5-dx*.14,ex=.5+dx*.14;
 return <svg className="camera-route turn-cue" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label={`Turn in place ${Math.round(degrees)} degrees ${direction}`}>
  <path d={`M ${sx} .9 A .14 .1 0 1 ${sweep} ${ex} .9`} className="turn-arc"/>
  <polygon points={`${ex-dx*.028},.875 ${ex+dx*.012},.9 ${ex-dx*.028},.93`} className="turn-head"/>
 </svg>;
}
