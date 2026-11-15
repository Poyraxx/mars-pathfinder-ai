import * as C from 'cesium';

const TIME=new C.JulianDate(0,0);
export function keepReplayDatum(scene:C.Scene,minimum:()=>number|null){
 return scene.preRender.addEventListener(()=>{
  const state=(scene as unknown as {frameState?:{minimumTerrainHeight:number}}).frameState;
  const floor=minimum();
  if(state&&floor!==null&&Number.isFinite(floor))state.minimumTerrainHeight=Math.min(state.minimumTerrainHeight,floor);
 });
}
export function updateReplayLine(entity:C.Entity,positions:C.Cartesian3[],clampToGround?:boolean){
 positions=positions.filter((p,i)=>!i||!C.Cartesian3.equalsEpsilon(p,positions[i-1],0,1e-6));
 if(positions.length<2)positions=[];
 const line=entity.polyline!;
 let changed=false;
 if(clampToGround!==undefined&&line.clampToGround?.getValue(TIME)!==clampToGround){line.clampToGround=new C.ConstantProperty(clampToGround);changed=true;}
 const previous=line.positions?.getValue(TIME) as C.Cartesian3[]|undefined;
 if(previous?.length===positions.length&&previous.every((p,i)=>C.Cartesian3.equalsEpsilon(p,positions[i],0,1e-6)))return changed;
 line.positions=new C.ConstantProperty(positions);return true;
}

export function measuredReplayLine(points:{lat:number;lon:number}[],heightAt:(lat:number,lon:number)=>number|null,ellipsoid:C.Ellipsoid){
 const sampled:{lat:number;lon:number}[]=[];
 points.forEach((p,i)=>{
  const prev=points[Math.max(0,i-1)];
  const metres=Math.hypot(p.lat-prev.lat,(p.lon-prev.lon)*Math.cos(C.Math.toRadians(p.lat)))*Math.PI/180*ellipsoid.maximumRadius;
  const n=Math.max(1,Math.ceil(metres/.75));
  for(let j=i?1:0;j<=n;j++)sampled.push({lat:C.Math.lerp(prev.lat,p.lat,j/n),lon:C.Math.lerp(prev.lon,p.lon,j/n)});
 });
 const heights=sampled.map(p=>heightAt(p.lat,p.lon));
 if(heights.some(h=>h===null||!Number.isFinite(h)))return {positions:C.Cartesian3.fromDegreesArray(sampled.flatMap(p=>[p.lon,p.lat]),ellipsoid),clampToGround:true};
 return {positions:C.Cartesian3.fromDegreesArrayHeights(sampled.flatMap((p,i)=>[p.lon,p.lat,heights[i]!+.15]),ellipsoid),clampToGround:false};
}

export function updateReplayPose(entity:C.Entity,position:C.Cartesian3,orientation?:C.Quaternion){
 let changed=false;
 if(!C.Cartesian3.equalsEpsilon(entity.position?.getValue(TIME),position,0,1e-6)){
  entity.position=new C.ConstantPositionProperty(position);changed=true;
 }
 if(orientation&&!C.Quaternion.equalsEpsilon(entity.orientation?.getValue(TIME),orientation,1e-10)){
  entity.orientation=new C.ConstantProperty(orientation);changed=true;
 }
 return changed;
}

export function updateReplayColour(entity:C.Entity,colour:C.Color){
 const material=entity.polyline!.material as C.PolylineGlowMaterialProperty;
 if(C.Color.equals(material.color?.getValue(TIME),colour))return false;
 material.color=new C.ConstantProperty(colour);return true;
}
