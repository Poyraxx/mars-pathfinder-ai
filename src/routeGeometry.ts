/**
 * One continuous, ground-painted guidance ribbon in image coordinates (0..1).
 *
 * Input: the projected ground ribbon quads [l0,r0,r1,l1] from the local planner (metric 0.5 m wide,
 * already in perspective) and, per quad, its ground arc length in metres along the path.
 * Output: a single outline polygon (no internal seams) whose arrow head covers the last
 * HEAD_M metres of ground, and chevrons painted every CHEVRON_EVERY_M metres. Because both are
 * defined on the ground, they shrink and foreshorten with distance exactly as the surface does.
 * Gaps where a nearer rise hides the ground are bridged so the guidance reads as one path; the
 * corridor itself remains computed from measured stereo only.
 */
export type P=[number,number];
export const ANCHOR:P=[.5,.985];
/** Painted line width as a fraction of the planner's 0.5 m ribbon (perspective preserved). */
export const PAINT=.55;
const HEAD_M=1.0,CHEVRON_EVERY_M=.9,CHEVRON_LEN_M=.28;

const mid=(a:P,b:P):P=>[(a[0]+b[0])/2,(a[1]+b[1])/2];
const dist=(a:P,b:P)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
const unit=(v:P):P=>{const n=Math.hypot(v[0],v[1])||1;return [v[0]/n,v[1]/n];};
const lerp=(a:number,b:number,t:number)=>a+(b-a)*t;

type Track={c:P[];w:number[];g:number[]};

/** Centre points, half-widths and ground metres of the projected ribbon, strips concatenated in order. */
function centreline(ribbons:number[][][],groundM?:number[][]):Track{
 const c:P[]=[],w:number[]=[],g:number[]=[];
 ribbons.forEach((q,i)=>{
  const l0=q[0] as P,r0=q[1] as P,r1=q[2] as P,l1=q[3] as P;
  const a=mid(l0,r0),b=mid(l1,r1),s=groundM?.[i]??[NaN,NaN];
  if(!c.length||dist(c[c.length-1],a)>1e-4){c.push(a);w.push(dist(l0,r0)/2);g.push(s[0]);}
  c.push(b);w.push(dist(l1,r1)/2);g.push(s[1]);
 });
 return {c,w,g};
}

/** Uniform resampling along image arc length; width and ground metres are interpolated. */
function resample({c,w,g}:Track,step:number):Track{
 const oc:P[]=[c[0]],ow=[w[0]],og=[g[0]];let carry=0;
 for(let i=1;i<c.length;i++){
  const seg=dist(c[i-1],c[i]);if(seg<1e-9)continue;
  let t=step-carry;
  while(t<=seg){const f=t/seg;oc.push([lerp(c[i-1][0],c[i][0],f),lerp(c[i-1][1],c[i][1],f)]);ow.push(lerp(w[i-1],w[i],f));og.push(lerp(g[i-1],g[i],f));t+=step;}
  carry=seg-(t-step);
 }
 const last=c.length-1;if(dist(oc[oc.length-1],c[last])>step*.25){oc.push(c[last]);ow.push(w[last]);og.push(g[last]);}
 return {c:oc,w:ow,g:og};
}

function smooth<T extends number|P>(v:T[],k:number,lockEnds=true):T[]{
 return v.map((_,i)=>{
  if(lockEnds&&(i===0||i===v.length-1))return v[i];
  const lo=Math.max(0,i-k),hi=Math.min(v.length-1,i+k);let n=0,x=0,y=0;
  for(let j=lo;j<=hi;j++){const p=v[j];if(typeof p==='number')x+=p;else{x+=p[0];y+=p[1];}n++;}
  return (typeof v[i]==='number'?x/n:[x/n,y/n]) as T;
 });
}

export type GroundArrow={outline:P[];chevrons:P[][];length:number;tip:P};

export function groundArrow(ribbons:number[][][],groundM?:number[][],head=true):GroundArrow|null{
 if(!ribbons.length)return null;
 let tr=centreline(ribbons,groundM?.length===ribbons.length?groundM:undefined);
 if(tr.c.length<2)return null;
 const metric=tr.g.every(Number.isFinite);
 // Link from the rover (bottom centre) to the first measured ground point: a cubic tangent to the route.
 // Its ground metres run up to the first projected point (the ground just ahead of the body).
 const {c:c0,w:w0,g:g0}=tr,k=Math.min(4,c0.length-1),dir=unit([c0[k][0]-c0[0][0],c0[k][1]-c0[0][1]]),d=dist(ANCHOR,c0[0]);
 const startW=Math.max(w0[0],.03/PAINT);
 if(d>.01){
  const p1:P=[ANCHOR[0],ANCHOR[1]-d*.45],p2:P=[c0[0][0]-dir[0]*d*.45,c0[0][1]-dir[1]*d*.45],lc:P[]=[],lw:number[]=[],lg:number[]=[];
  for(let i=0;i<24;i++){const t=i/24,u=1-t;
   lc.push([u*u*u*ANCHOR[0]+3*u*u*t*p1[0]+3*u*t*t*p2[0]+t*t*t*c0[0][0],u*u*u*ANCHOR[1]+3*u*u*t*p1[1]+3*u*t*t*p2[1]+t*t*t*c0[0][1]]);
   lw.push(lerp(startW,w0[0],t));lg.push(metric?Math.max(0,g0[0]-.8*(1-t)):NaN);}
  tr={c:[...lc,...c0],w:[...lw,...w0],g:[...lg,...g0]};
 }
 tr=resample(tr,.006);
 const c=smooth(tr.c,3),g=tr.g;let w=tr.w;
 // Perspective: ground width can only shrink away from the camera.
 for(let i=1;i<w.length;i++)w[i]=Math.min(w[i],w[i-1]);
 w=smooth(w,4,false).map(x=>Math.max(.004,x*PAINT));
 const s=[0];for(let i=1;i<c.length;i++)s.push(s[i-1]+dist(c[i-1],c[i]));
 const total=s[s.length-1];if(total<.03)return null;
 const tan=c.map((_,i)=>unit([c[Math.min(i+2,c.length-1)][0]-c[Math.max(i-2,0)][0],c[Math.min(i+2,c.length-1)][1]-c[Math.max(i-2,0)][1]]));
 const gEnd=g[g.length-1];
 // Arrow head: the last HEAD_M metres of ground (foreshortened by the camera), base 1.8x the local width.
 let b=c.length-1;
 if(!head){}
 else if(metric)while(b>0&&gEnd-g[b]<HEAD_M&&total-s[b]<total*.4)b--;
 else{const headLen=Math.min(total*.3,w[w.length-1]*4.6);while(b>0&&total-s[b]<headLen)b--;}
 const left:P[]=[],right:P[]=[];
 for(let i=0;i<=b;i++){const n:P=[-tan[i][1],tan[i][0]];left.push([c[i][0]+n[0]*w[i],c[i][1]+n[1]*w[i]]);right.push([c[i][0]-n[0]*w[i],c[i][1]-n[1]*w[i]]);}
 const tip=c[c.length-1],hw=w[b]*2.0,ht=unit([tip[0]-c[b][0],tip[1]-c[b][1]]),nh:P=[-ht[1],ht[0]];
 // Without a head the ribbon simply ends on the ground; the plan continues as a thin line beyond it.
 const outline=head?[...left,[c[b][0]+nh[0]*hw,c[b][1]+nh[1]*hw] as P,tip,[c[b][0]-nh[0]*hw,c[b][1]-nh[1]*hw] as P,...right.reverse()]:[...left,...right.reverse()];
 // Chevrons painted on the surface: every CHEVRON_EVERY_M metres of ground, CHEVRON_LEN_M long on the ground,
 // so their spacing and length foreshorten with distance; their width follows the local ribbon width.
 const chevrons:P[][]=[];
 const chevron=(i:number,len:number)=>{const n:P=[-tan[i][1],tan[i][0]],ww=w[i]*.62;
  const apex:P=[c[i][0]+tan[i][0]*len*.5,c[i][1]+tan[i][1]*len*.5],back:P=[c[i][0]-tan[i][0]*len*.5,c[i][1]-tan[i][1]*len*.5];
  chevrons.push([[back[0]+n[0]*ww,back[1]+n[1]*ww],apex,[back[0]-n[0]*ww,back[1]-n[1]*ww]]);};
 if(metric){
  let next=Math.ceil((g[0]+.4)/CHEVRON_EVERY_M)*CHEVRON_EVERY_M;
  for(let i=1;i<b;i++){
   if(g[i]<next)continue;
   const j=Math.min(i+3,c.length-1),dg=Math.max(1e-4,g[j]-g[Math.max(i-3,0)]),ds=s[j]-s[Math.max(i-3,0)];
   const len=Math.min(w[i]*1.4,ds/dg*CHEVRON_LEN_M);
   if(gEnd-g[i]>HEAD_M*.6)chevron(i,len);
   next+=CHEVRON_EVERY_M;while(next<=g[i])next+=CHEVRON_EVERY_M;
  }
 }else{
  let next=startW*PAINT*1.6;
  for(let i=0;i<b;i++){if(s[i]<next)continue;chevron(i,w[i]*.7);next=s[i]+w[i]*4.2;}
 }
 return {outline,chevrons,length:total,tip};
}
