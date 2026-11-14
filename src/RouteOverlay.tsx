import type {LocalNavigation} from './types';

type P=[number,number];
const ANCHOR:P=[.5,.985];
const ANCHOR_HALF=.05;

/** Join consecutive projected ribbon quads ([l0,r0,r1,l1]) into continuous strips. */
function strips(ribbons:number[][][]):{left:P[];right:P[]}[]{
 const out:{left:P[];right:P[]}[]=[];
 let cur:{left:P[];right:P[]}|null=null;
 for(const q of ribbons){
  const l0=q[0] as P,r0=q[1] as P,r1=q[2] as P,l1=q[3] as P;
  const joined=cur&&Math.hypot(cur.left[cur.left.length-1][0]-l0[0],cur.left[cur.left.length-1][1]-l0[1])<.004;
  if(!joined){cur={left:[l0],right:[r0]};out.push(cur);}
  cur!.left.push(l1);cur!.right.push(r1);
 }
 return out.filter(s=>s.left.length>=2);
}

const mid=(a:P,b:P):P=>[(a[0]+b[0])/2,(a[1]+b[1])/2];
const norm=(v:P):P=>{const n=Math.hypot(v[0],v[1])||1;return [v[0]/n,v[1]/n];};

/** Image-plane link from the rover (bottom centre) to the first visible ground point, tangent to the route. */
function connector(first:{left:P[];right:P[]}):P[]{
 const m0=mid(first.left[0],first.right[0]);
 const m1=mid(first.left[Math.min(3,first.left.length-1)],first.right[Math.min(3,first.right.length-1)]);
 const dir=norm([m1[0]-m0[0],m1[1]-m0[1]]);
 const dist=Math.hypot(m0[0]-ANCHOR[0],m0[1]-ANCHOR[1]);
 const p1:P=[ANCHOR[0],ANCHOR[1]-dist*.45],p2:P=[m0[0]-dir[0]*dist*.45,m0[1]-dir[1]*dist*.45];
 const w1=Math.hypot(first.left[0][0]-first.right[0][0],first.left[0][1]-first.right[0][1])/2;
 const left:P[]=[],right:P[]=[];
 for(let i=0;i<=28;i++){
  const t=i/28,u=1-t;
  const x=u*u*u*ANCHOR[0]+3*u*u*t*p1[0]+3*u*t*t*p2[0]+t*t*t*m0[0];
  const y=u*u*u*ANCHOR[1]+3*u*u*t*p1[1]+3*u*t*t*p2[1]+t*t*t*m0[1];
  const dx=3*u*u*(p1[0]-ANCHOR[0])+6*u*t*(p2[0]-p1[0])+3*t*t*(m0[0]-p2[0]);
  const dy=3*u*u*(p1[1]-ANCHOR[1])+6*u*t*(p2[1]-p1[1])+3*t*t*(m0[1]-p2[1]);
  const n=norm([-dy,dx]),w=ANCHOR_HALF*(1-t)+w1*t;
  left.push([x+n[0]*w,y+n[1]*w]);right.push([x-n[0]*w,y-n[1]*w]);
 }
 return [...left,...right.reverse()];
}

const pts=(ps:P[])=>ps.map(p=>`${p[0].toFixed(4)},${p[1].toFixed(4)}`).join(' ');

export default function RouteOverlay({local,faint}:{local:LocalNavigation;faint:boolean}){
 const parts=strips(local.ribbons);
 const opacity=faint?.4:.88;
 const last=parts[parts.length-1];
 let arrow:P[]|null=null;
 if(last&&last.left.length>=3){
  const n=last.left.length-1,tip=mid(last.left[n],last.right[n]),prev=mid(last.left[n-2],last.right[n-2]);
  const d=norm([tip[0]-prev[0],tip[1]-prev[1]]),w=Math.max(.012,Math.hypot(last.left[n][0]-last.right[n][0],last.left[n][1]-last.right[n][1])*1.1);
  const nrm:P=[-d[1],d[0]];
  arrow=[[tip[0]+d[0]*w*1.6,tip[1]+d[1]*w*1.6],[tip[0]+nrm[0]*w,tip[1]+nrm[1]*w],[tip[0]-nrm[0]*w,tip[1]-nrm[1]*w]];
 }
 return <svg className="camera-route route-overlay" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Rover route: link from the rover, then the collision-checked ground route">
  {(local.corridor_edges??[]).map((line,i)=><polyline key={`c${i}`} className="corridor-edge" points={line.map(p=>`${p[0]},${p[1]}`).join(' ')}/>)}
  {parts[0]&&<polygon className="route-link" points={pts(connector(parts[0]))} fillOpacity={opacity*.85}/>}
  {parts.map((s,i)=><polygon key={i} className="route-strip" points={pts([...s.left,...[...s.right].reverse()])} fillOpacity={opacity}/>)}
  {arrow&&<polygon className="route-strip" points={pts(arrow)} fillOpacity={opacity}/>}
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
