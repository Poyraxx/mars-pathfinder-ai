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

export type GroundArrow={outline:P[];chevrons:P[][];length:number;tip:P;centre:P[];width:number[]};

export function groundArrow(ribbons:number[][][],groundM?:number[][],head=true):GroundArrow|null{
 if(!ribbons.length)return null;
 let tr=centreline(ribbons,groundM?.length===ribbons.length?groundM:undefined);
 if(tr.c.length<2)return null;
 const metric=tr.g.every(Number.isFinite);
 // The unmeasured image-space connection is a direct bearing to measured ground.
 // A tangent-matching cubic used to add an opposite-side sweep before the
 // real bypass, making one manoeuvre look like left/right/left instructions.
 // Its ground metres run up to the first projected point (the ground just ahead of the body).
 const {c:c0,w:w0,g:g0}=tr,d=dist(ANCHOR,c0[0]);
 const startW=Math.max(w0[0],.03/PAINT);
 if(d>.01){
  const lc:P[]=[],lw:number[]=[],lg:number[]=[];
  for(let i=0;i<24;i++){const t=i/24;
   lc.push([lerp(ANCHOR[0],c0[0][0],t),lerp(ANCHOR[1],c0[0][1],t)]);
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
 return {outline,chevrons,length:total,tip,centre:c,width:w};
}

/** Projected map context, distinct from the collision-checked local corridor. */
export function mapRoute(segments:number[][][],fromM=2){
 const extent=Math.max(0,...segments.flatMap(seg=>seg.map(p=>p[2]??0)));
 // A goal close to the front wheels can have less than two metres of projected
 // remaining path. Keep its actual visible ground rather than dropping it all.
 const closeGoal=fromM<=2&&extent>0&&extent<fromM;
 const threshold=closeGoal?extent*.5:fromM;
 const visible=segments.map(seg=>{
  const valid=seg.filter(p=>p.length>=3&&p.every(Number.isFinite));
  const clipped=valid.filter(p=>p[2]>=threshold);
  // Retain a two-point final fragment on sparse short plans. Earlier body-side
  // points must not pull the centre guide backwards into a hook at the rover.
  return closeGoal&&clipped.length===1&&valid.at(-1)?.[2]===extent?valid.slice(-2):clipped;
 })
  .filter(s=>s.length>=2).map(seg=>seg.map((p,i)=>{
   if(i===0||i===seg.length-1)return p;
   const half=Math.max(1.5,p[2]*.06);let x=0,y=0,n=0;
   for(let j=i;j>=0&&p[2]-seg[j][2]<=half;j--){x+=seg[j][0];y+=seg[j][1];n++;}
   for(let j=i+1;j<seg.length&&seg[j][2]-p[2]<=half;j++){x+=seg[j][0];y+=seg[j][1];n++;}
   return [x/n,y/n,p[2]];
  })).filter(s=>s.reduce((v,p,i)=>v+(i?dist(p as P,s[i-1] as P):0),0)>=.008);
 const hidden:P[][]=[];
 for(let i=1;i<visible.length;i++){
  const a=visible[i-1].at(-1)!,b=visible[i][0];
  if(dist(a as P,b as P)<.15)hidden.push([[a[0],a[1]],[b[0],b[1]]]);
 }
 return {visible,hidden};
}

export type RibbonStroke={outline:P[];centre:P[];width:number[];planned:P[]};
export type NavigationRibbon={strokes:RibbonStroke[];chevrons:P[][];hidden:P[][];end:P|null;hasLocal:boolean};

function outlineTrack(centre:P[],width:number[],planned:P[]):RibbonStroke{
 const left:P[]=[],right:P[]=[];
 for(let i=0;i<centre.length;i++){
  const a=centre[Math.max(0,i-1)],b=centre[Math.min(centre.length-1,i+1)],t=unit([b[0]-a[0],b[1]-a[1]]);
  left.push([centre[i][0]-t[1]*width[i],centre[i][1]+t[0]*width[i]]);
  right.push([centre[i][0]+t[1]*width[i],centre[i][1]-t[0]*width[i]]);
 }
 return {outline:[...left,...right.reverse()],centre,width,planned};
}

/** One tapering outline across the near/map handoff. The planned continuation
 * remains marked separately; joining display geometry does not validate terrain.
 * True camera exits/occlusions remain separate strips, never invented ground. */
export function navigationRibbon(ribbons:number[][][],groundM:number[][]|undefined,segments:number[][][],localLength=0,planningLink=false,manoeuvre=false):NavigationRibbon{
 const near=groundArrow(ribbons,groundM,false),plan=mapRoute(segments,near?Math.max(2,localLength+.5):2);
 let visible=plan.visible,attachNear=false;
 if(near&&visible.length){
  // The first projected map fragment can lie behind the local endpoint after a
  // detour. Attach to the nearest forward fragment, not back toward the rover.
  const before=near.centre[Math.max(0,near.centre.length-4)],tangent=unit([near.tip[0]-before[0],near.tip[1]-before[1]]);
  let best=Infinity,attachment:[number,number]|null=null;
  for(let si=0;si<visible.length;si++)for(let pi=0;pi<visible[si].length;pi++){
   const p=visible[si][pi];
   const delta:P=[p[0]-near.tip[0],p[1]-near.tip[1]],d=Math.hypot(...delta);
   if(p[1]>near.tip[1]+.025||delta[0]*tangent[0]+delta[1]*tangent[1]<-.03)continue;
   const score=d+p[2]*.00001;
   if(score<best){best=score;attachment=[si,pi];}
  }
  if(attachment&&best<.45){
   const [si,pi]=attachment;visible=visible.slice(si);visible[0]=visible[0].slice(pi);attachNear=true;
  }
 }
 const strokes:RibbonStroke[]=[],chevrons=near?.chevrons.map(ch=>[...ch])??[];
 let c:P[]=near?[...near.centre]:[],w=near?[...near.width]:[],planned:P[]=[];
 if(!near&&planningLink&&visible.length){
  // Centre-guide mode may connect the projected map to the rover as a thin HUD
  // direction cue. Every point is planned/unchecked; hasLocal stays false.
  const first=visible[0][0],end:P=[first[0],first[1]];
  for(let j=0;j<=32;j++){
   const t=j/32;
   const point:P=[lerp(ANCHOR[0],end[0],t),lerp(ANCHOR[1],end[1],t)];
   c.push(point);w.push(lerp(.006,.0012+.004*end[1]**2,t*t*(3-2*t)));planned.push(point);
  }
  attachNear=true;
 }
 if(near&&(manoeuvre||(visible.length&&dist(near.tip,visible[0][0] as P)<.45))){
  // Spread the paint taper over the whole visible approach. A fixed short
  // end-window pinched a broad middle into a needle just before the map join.
  // This changes paint only; the measured centre and clearance stay unchanged.
  const target=Math.min(w.at(-1)!, .0012+.004*(visible[0]?.[0][1]??near.tip[1])**2);
  const arc=[0];for(let i=1;i<c.length;i++)arc.push(arc[i-1]+dist(c[i-1],c[i]));
  const total=arc.at(-1)||1;
  for(let i=0;i<c.length;i++){
   const t=arc[i]/total,blend=t*t*(3-2*t);
   w[i]=lerp(w[i],target,blend);
  }
  chevrons.forEach((ch,index)=>{
   const at:P=[ch.reduce((v,p)=>v+p[0],0)/ch.length,ch.reduce((v,p)=>v+p[1],0)/ch.length];
   let i=0;for(let j=1;j<c.length;j++)if(dist(c[j],at)<dist(c[i],at))i=j;
   const scale=w[i]/near.width[i];
   chevrons[index]=ch.map(p=>[at[0]+(p[0]-at[0])*scale,at[1]+(p[1]-at[1])*scale]);
  });
 }
 const flush=()=>{if(c.length>=2)strokes.push(outlineTrack(c,w,planned));c=[];w=[];planned=[];};
 for(const seg of visible){
  const first:P=[seg[0][0],seg[0][1]],last=c.at(-1),joined=attachNear&&!!last&&dist(last,first)<.45&&strokes.length===0;
  let extension:P[]=[];const joinWidth=joined?w.at(-1)!:0;
  if(joined&&last){
   const before=c[Math.max(0,c.length-4)],after=seg[Math.min(3,seg.length-1)],d=dist(last,first);
   const a=unit([last[0]-before[0],last[1]-before[1]]),b=unit([after[0]-first[0],after[1]-first[1]]);
   const p1:P=[last[0]+a[0]*d*.3,last[1]+a[1]*d*.3],p2:P=[first[0]-b[0]*d*.3,first[1]-b[1]*d*.3];
   // A display handoff must not overshoot either endpoint and double back.
   for(const axis of [0,1] as const){
    p1[axis]=Math.max(Math.min(last[axis],first[axis]),Math.min(Math.max(last[axis],first[axis]),p1[axis]));
    p2[axis]=Math.max(Math.min(p1[axis],first[axis]),Math.min(Math.max(p1[axis],first[axis]),p2[axis]));
   }
   if(!planned.length||dist(planned.at(-1)!,last)>1e-7)planned.push(last);
   for(let j=1;j<24&&d>.0001;j++){
    const t=j/24,u=1-t;
    extension.push([u*u*u*last[0]+3*u*u*t*p1[0]+3*u*t*t*p2[0]+t*t*t*first[0],u*u*u*last[1]+3*u*u*t*p1[1]+3*u*t*t*p2[1]+t*t*t*first[1]]);
   }
  }else flush();
  extension.push(...seg.map(p=>[p[0],p[1]] as P));
  let travelled=0,previous=joined?last:extension[0];
  const taperDistance=joined?Math.max(.07,Math.min(.25,dist(last!,first)+.06)):.07;
  for(const p of extension){
   travelled+=previous?dist(previous,p):0;previous=p;
   const base=.0012+.004*Math.max(0,Math.min(1,p[1]))**2;
   const t=Math.min(1,travelled/taperDistance),blend=t*t*(3-2*t);
   const half=joined?lerp(joinWidth,Math.min(joinWidth,base),blend):base;
   if(c.length&&dist(c.at(-1)!,p)<1e-7)continue;
   c.push(p);w.push(half);planned.push(p);
  }
  // Subsequent visible pieces are separated by a true camera/occlusion gap.
  flush();
 }
 flush();
 const hidden:P[][]=[];
 for(let i=1;i<visible.length;i++){
  const a=visible[i-1].at(-1)!,b=visible[i][0];
  if(dist(a as P,b as P)<.15)hidden.push([[a[0],a[1]],[b[0],b[1]]]);
 }
 return {strokes,chevrons,hidden,end:strokes.at(-1)?.centre.at(-1)??null,hasLocal:!!near};
}
