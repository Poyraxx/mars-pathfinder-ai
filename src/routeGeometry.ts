/**
 * One continuous, ground-painted guidance ribbon in image coordinates (0..1).
 *
 * Input: the projected ground ribbon quads [l0,r0,r1,l1] from the local planner (metric 0.5 m wide,
 * already in perspective). Output: a single outline polygon (no internal seams), an arrow head that is
 * part of the same outline, and chevrons painted along the surface. Width follows perspective: it can
 * only shrink with distance, so the ribbon never "suddenly thins" and widens again. Gaps where a nearer
 * rise hides the ground are bridged so the guidance reads as one path; the corridor remains computed
 * from measured stereo only.
 */
export type P=[number,number];
export const ANCHOR:P=[.5,.985];
/** Painted line width as a fraction of the planner's 0.5 m ribbon (perspective preserved). */
export const PAINT=.55;

const mid=(a:P,b:P):P=>[(a[0]+b[0])/2,(a[1]+b[1])/2];
const dist=(a:P,b:P)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
const unit=(v:P):P=>{const n=Math.hypot(v[0],v[1])||1;return [v[0]/n,v[1]/n];};

/** Centre points and half-widths of the projected ribbon, strips concatenated in order. */
function centreline(ribbons:number[][][]):{c:P[];w:number[]}{
 const c:P[]=[],w:number[]=[];
 for(const q of ribbons){
  const l0=q[0] as P,r0=q[1] as P,r1=q[2] as P,l1=q[3] as P;
  const a=mid(l0,r0),b=mid(l1,r1);
  if(!c.length||dist(c[c.length-1],a)>1e-4){c.push(a);w.push(dist(l0,r0)/2);}
  c.push(b);w.push(dist(l1,r1)/2);
 }
 return {c,w};
}

/** Uniform resampling along arc length. */
function resample(c:P[],w:number[],step:number):{c:P[];w:number[]}{
 const oc:P[]=[c[0]],ow:number[]=[w[0]];let carry=0;
 for(let i=1;i<c.length;i++){
  const seg=dist(c[i-1],c[i]);if(seg<1e-9)continue;
  let t=step-carry;
  while(t<=seg){const f=t/seg;oc.push([c[i-1][0]+(c[i][0]-c[i-1][0])*f,c[i-1][1]+(c[i][1]-c[i-1][1])*f]);ow.push(w[i-1]+(w[i]-w[i-1])*f);t+=step;}
  carry=seg-(t-step);
 }
 const last=c[c.length-1];if(dist(oc[oc.length-1],last)>step*.25){oc.push(last);ow.push(w[w.length-1]);}
 return {c:oc,w:ow};
}

function smooth<T extends number|P>(v:T[],k:number,lockEnds=true):T[]{
 return v.map((_,i)=>{
  if(lockEnds&&(i===0||i===v.length-1))return v[i];
  const lo=Math.max(0,i-k),hi=Math.min(v.length-1,i+k);let n=0,x=0,y=0;
  for(let j=lo;j<=hi;j++){const p=v[j];if(typeof p==='number')x+=p;else{x+=p[0];y+=p[1];}n++;}
  return (typeof v[i]==='number'?x/n:[x/n,y/n]) as T;
 });
}

export type GroundArrow={outline:P[];chevrons:P[][];length:number};

export function groundArrow(ribbons:number[][][]):GroundArrow|null{
 if(!ribbons.length)return null;
 let {c,w}=centreline(ribbons);
 if(c.length<2)return null;
 // Link from the rover (bottom centre) to the first measured ground point: a cubic tangent to the route.
 const k=Math.min(4,c.length-1),dir=unit([c[k][0]-c[0][0],c[k][1]-c[0][1]]),d=dist(ANCHOR,c[0]);
 const startW=Math.max(w[0],.03/PAINT);
 if(d>.01){
  const p1:P=[ANCHOR[0],ANCHOR[1]-d*.45],p2:P=[c[0][0]-dir[0]*d*.45,c[0][1]-dir[1]*d*.45],lc:P[]=[],lw:number[]=[];
  for(let i=0;i<24;i++){const t=i/24,u=1-t;
   lc.push([u*u*u*ANCHOR[0]+3*u*u*t*p1[0]+3*u*t*t*p2[0]+t*t*t*c[0][0],u*u*u*ANCHOR[1]+3*u*u*t*p1[1]+3*u*t*t*p2[1]+t*t*t*c[0][1]]);
   lw.push(startW+(w[0]-startW)*t);}
  c=[...lc,...c];w=[...lw,...w];
 }
 ({c,w}=resample(c,w,.006));
 c=smooth(c,3);
 // Perspective: ground width can only shrink away from the camera.
 for(let i=1;i<w.length;i++)w[i]=Math.min(w[i],w[i-1]);
 w=smooth(w,4,false).map(x=>Math.max(.005,x*PAINT));
 // Arc length and tangents.
 const s=[0];for(let i=1;i<c.length;i++)s.push(s[i-1]+dist(c[i-1],c[i]));
 const total=s[s.length-1];if(total<.03)return null;
 const tan=c.map((_,i)=>unit([c[Math.min(i+2,c.length-1)][0]-c[Math.max(i-2,0)][0],c[Math.min(i+2,c.length-1)][1]-c[Math.max(i-2,0)][1]]));
 // Arrow head: occupies the last ~3.2 local widths of the path, base 2.1x the ribbon width.
 const tipW=w[w.length-1],headLen=Math.min(total*.3,tipW*4.6);
 let b=c.length-1;while(b>0&&total-s[b]<headLen)b--;
 const left:P[]=[],right:P[]=[];
 for(let i=0;i<=b;i++){const n:P=[-tan[i][1],tan[i][0]];left.push([c[i][0]+n[0]*w[i],c[i][1]+n[1]*w[i]]);right.push([c[i][0]-n[0]*w[i],c[i][1]-n[1]*w[i]]);}
 const nb:P=[-tan[b][1],tan[b][0]],hw=w[b]*1.8,tip=c[c.length-1];
 const headTan=unit([tip[0]-c[b][0],tip[1]-c[b][1]]),nh:P=[-headTan[1],headTan[0]];
 const wingL:P=[c[b][0]+nh[0]*hw,c[b][1]+nh[1]*hw],wingR:P=[c[b][0]-nh[0]*hw,c[b][1]-nh[1]*hw];
 void nb;
 const outline=[...left,wingL,tip,wingR,...right.reverse()];
 // Chevrons painted on the surface: spacing and size scale with the local (perspective) width.
 const chevrons:P[][]=[];let next=startW*PAINT*1.6;
 for(let i=0;i<b;i++){
  if(s[i]<next)continue;
  const n:P=[-tan[i][1],tan[i][0]],ww=w[i]*.62,len=w[i]*.7;
  const apex:P=[c[i][0]+tan[i][0]*len*.5,c[i][1]+tan[i][1]*len*.5];
  const back:P=[c[i][0]-tan[i][0]*len*.5,c[i][1]-tan[i][1]*len*.5];
  chevrons.push([[back[0]+n[0]*ww,back[1]+n[1]*ww],apex,[back[0]-n[0]*ww,back[1]-n[1]*ww]]);
  next=s[i]+w[i]*4.2;
 }
 return {outline,chevrons,length:total};
}
