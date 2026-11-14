import type {LocalNavigation} from './types';
import {groundArrow} from './routeGeometry';

type P=[number,number];
const pts=(ps:P[])=>ps.map(p=>`${p[0].toFixed(4)},${p[1].toFixed(4)}`).join(' ');
const CHEVRON_EVERY_M=12;

/** Far part of the planned route: the remaining map route projected with the PDS camera model onto the DEM.
 * Only points beyond the measured local corridor are drawn, as a thin line with small direction marks and a
 * dot at the last visible point. Display smoothing is along the route in metres (window grows with distance,
 * removing the 1 m planning-grid staircase); endpoints are kept. Hidden stretches between nearby visible
 * pieces are dotted; tiny or far-apart fragments (route leaving the frame) are not joined. */
function farRoute(segments:number[][][],fromM:number){
 const smoothM=(seg:number[][])=>seg.map((p,i)=>{
  if(i===0||i===seg.length-1)return p;
  const half=Math.max(1.5,p[2]*.06);let x=0,y=0,n=0;
  for(let j=i;j>=0&&p[2]-seg[j][2]<=half;j--){x+=seg[j][0];y+=seg[j][1];n++;}
  for(let j=i+1;j<seg.length&&seg[j][2]-p[2]<=half;j++){x+=seg[j][0];y+=seg[j][1];n++;}
  return [x/n,y/n,p[2]];});
 const len=(seg:number[][])=>seg.reduce((t,p,i)=>i?t+Math.hypot(p[0]-seg[i-1][0],p[1]-seg[i-1][1]):0,0);
 const visible=segments.map(s=>s.filter(p=>p[2]>=fromM)).filter(s=>s.length>=4).map(smoothM).filter(s=>len(s)>=.02);
 const marks:{at:P;dir:P;size:number}[]=[];let lastMark:P|null=null;
 for(const seg of visible){
  let next=Math.ceil((seg[0][2]+4)/CHEVRON_EVERY_M)*CHEVRON_EVERY_M;
  for(let i=1;i<seg.length-1;i++){
   if(seg[i][2]<next)continue;
   next+=CHEVRON_EVERY_M;
   const at:P=[seg[i][0],seg[i][1]];
   // Keep marks readable near the horizon: never closer than 4 % of the image to the previous mark.
   if(lastMark&&Math.hypot(at[0]-lastMark[0],at[1]-lastMark[1])<.04)continue;
   const a=seg[Math.max(0,i-8)],b=seg[Math.min(seg.length-1,i+8)],d=Math.hypot(b[0]-a[0],b[1]-a[1])||1;
   marks.push({at,dir:[(b[0]-a[0])/d,(b[1]-a[1])/d],size:.005+.012*seg[i][1]});lastMark=at;
  }
 }
 const hidden:P[][]=[];
 for(let i=1;i<visible.length;i++){const a=visible[i-1][visible[i-1].length-1],b=visible[i][0];if(Math.hypot(a[0]-b[0],a[1]-b[1])<.15)hidden.push([[a[0],a[1]],[b[0],b[1]]]);}
 const last=visible.length?visible[visible.length-1][visible[visible.length-1].length-1]:null;
 return {visible,hidden,marks,first:visible[0]?.[0]??null,last};
}

/** Route painted on the ground near the rover, continuing as a thin planned line to the destination. */
export default function RouteOverlay({local,faint,far,farEndIsGoal}:{local:LocalNavigation|null;faint:boolean;far?:number[][][];farEndIsGoal?:boolean}){
 const arrow=local?.ribbons.length?groundArrow(local.ribbons,local.ribbon_s_m,false):null;
 const k=faint?.45:1;
 const top=arrow?Math.min(...arrow.outline.map(p=>p[1])):0;
 const plan=far?.length?farRoute(far,arrow?(local?.length_m??0)+1:2):null;
 const link=arrow&&plan?.first&&Math.hypot(arrow.tip[0]-plan.first[0],arrow.tip[1]-plan.first[1])<.45?[arrow.tip,[plan.first[0],plan.first[1]] as P]:null;
 const end:P|null=plan?.last?[plan.last[0],plan.last[1]]:arrow?arrow.tip:null;
 return <svg className="camera-route route-overlay" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Planned route: painted on the measured ground near the rover, then a thin line along the map plan to the destination">
  <defs>
   <linearGradient id="route-depth" gradientUnits="userSpaceOnUse" x1="0" y1="1" x2="0" y2={top}>
    <stop offset="0" stopColor="#3ce1ff" stopOpacity={.66*k}/>
    <stop offset=".55" stopColor="#3ce1ff" stopOpacity={.5*k}/>
    <stop offset="1" stopColor="#7cf0ff" stopOpacity={.34*k}/>
   </linearGradient>
   <filter id="route-glow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation=".006"/></filter>
  </defs>
  {(local?.corridor_edges??[]).map((line,i)=><polyline key={`c${i}`} className="corridor-edge" points={line.map(p=>`${p[0]},${p[1]}`).join(' ')}/>)}
  {arrow&&<>
   <polygon className="route-shadow" points={pts(arrow.outline)} filter="url(#route-glow)"/>
   <polygon className="route-paint" points={pts(arrow.outline)} fill="url(#route-depth)"/>
   {arrow.chevrons.map((c,i)=><polyline key={i} className="route-chevron" points={pts(c)} opacity={k}/>)}
  </>}
  {plan&&<g className="route-far" opacity={k}>
   {link&&<polyline className="route-far-hidden" points={pts(link)}/>}
   {plan.hidden.map((h,i)=><polyline key={`h${i}`} className="route-far-hidden" points={pts(h)}/>)}
   {plan.visible.map((s,i)=><polyline key={`o${i}`} className="route-far-outline" points={pts(s.map(p=>[p[0],p[1]] as P))}/>)}
   {plan.visible.map((s,i)=><polyline key={`v${i}`} className="route-far-line" points={pts(s.map(p=>[p[0],p[1]] as P))}/>)}
   {plan.marks.map((m,i)=>{const n:P=[-m.dir[1],m.dir[0]],t:P=[m.at[0]+m.dir[0]*m.size,m.at[1]+m.dir[1]*m.size];
    return <polygon key={`m${i}`} className="route-far-mark" points={pts([t,[m.at[0]-m.dir[0]*m.size*.6+n[0]*m.size*.75,m.at[1]-m.dir[1]*m.size*.6+n[1]*m.size*.75],[m.at[0]-m.dir[0]*m.size*.6-n[0]*m.size*.75,m.at[1]-m.dir[1]*m.size*.6-n[1]*m.size*.75]])}/>;})}
  </g>}
  {end&&<g className="route-end" opacity={k}><circle cx={end[0]} cy={end[1]} r=".011" className="route-end-halo"/><circle cx={end[0]} cy={end[1]} r=".0055" className="route-end-dot"/>{farEndIsGoal&&<title>Destination</title>}</g>}
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
