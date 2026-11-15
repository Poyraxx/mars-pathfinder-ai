import {test} from 'node:test';
import assert from 'node:assert/strict';
import * as C from 'cesium';
import {updateReplayLine,updateReplayPose,updateReplayColour,measuredReplayLine,keepReplayDatum} from '../src/replayScene.ts';

const time=new C.JulianDate(0,0);
const points=()=>[new C.Cartesian3(1,2,3),new C.Cartesian3(4,5,6)];
const line=()=>new C.Entity({polyline:{positions:[],clampToGround:true,material:new C.PolylineGlowMaterialProperty({color:C.Color.CYAN})}});

test('negative-elevation annotations survive a zero-height LOD bound without seeing through the far side',()=>{
 const camera=new C.Cartesian3(950,0,0),near=new C.BoundingSphere(new C.Cartesian3(900.15,0,0),.01),far=new C.BoundingSphere(new C.Cartesian3(-900.15,0,0),.01);
 const scene={preRender:new C.Event(),frameState:{minimumTerrainHeight:0}};
 const visible=sphere=>new C.Occluder(new C.BoundingSphere(C.Cartesian3.ZERO,1000+scene.frameState.minimumTerrainHeight),camera).isBoundingSphereVisible(sphere);
 assert.equal(visible(near),false);
 const release=keepReplayDatum(scene,()=>-100);
 for(let i=0;i<120;i++){
  scene.frameState.minimumTerrainHeight=0;scene.preRender.raiseEvent();
  assert.equal(visible(near),true);assert.equal(visible(far),false);
 }
 scene.frameState.minimumTerrainHeight=-150;scene.preRender.raiseEvent();
 assert.equal(scene.frameState.minimumTerrainHeight,-150);
 release();scene.frameState.minimumTerrainHeight=0;scene.preRender.raiseEvent();
 assert.equal(scene.frameState.minimumTerrainHeight,0);
});

test('paused ground geometry stays constant across render ticks and duplicate stationary poses',()=>{
 const e=line();assert.equal(updateReplayLine(e,points()),true);
 const property=e.polyline.positions;let changed=0;
 e.definitionChanged.addEventListener(()=>changed++);
 for(let i=0;i<120;i++){
  const p=points();p.push(C.Cartesian3.clone(p[1]));
  assert.equal(updateReplayLine(e,p),false);
  assert.equal(e.polyline.positions,property);
  assert.equal(e.polyline.positions.isConstant,true);
 }
 assert.equal(changed,0);
 const moved=points();moved[1].x+=.5;
 assert.equal(updateReplayLine(e,moved),true);
 assert.equal(changed,1);assert.equal(e.polyline.positions.getValue(time)[1].x,4.5);
});

test('sampled replay line follows measured heights without depending on terrain LOD',()=>{
 const mars=new C.Ellipsoid(3396190,3396190,3396190);
 const height=(lat,lon)=>-4517+lon*50000+lat*10000;
 const path=measuredReplayLine([{lat:0,lon:0},{lat:0,lon:.0001}],height,mars);
 assert.equal(path.clampToGround,false);assert.ok(path.positions.length>5);
 for(const p of path.positions){
  const geo=mars.cartesianToCartographic(p);
  assert.ok(Math.abs(geo.height-height(C.Math.toDegrees(geo.latitude),C.Math.toDegrees(geo.longitude))-.15)<1e-6);
 }
 const e=line();updateReplayLine(e,path.positions,path.clampToGround);
 const property=e.polyline.positions;
 for(let i=0;i<120;i++)assert.equal(updateReplayLine(e,path.positions.map(p=>C.Cartesian3.clone(p)),false),false);
 assert.equal(e.polyline.positions,property);assert.equal(e.polyline.clampToGround.getValue(time),false);
});

test('missing DEM coverage uses terrain clamping instead of made-up zero height',()=>{
 const path=measuredReplayLine([{lat:0,lon:0},{lat:0,lon:.0001}],()=>null,C.Ellipsoid.WGS84);
 assert.equal(path.clampToGround,true);
});

test('stationary rover preserves clamped position and orientation properties',()=>{
 const e=new C.Entity();const p=new C.Cartesian3(1,2,3),q=C.Quaternion.IDENTITY;
 assert.equal(updateReplayPose(e,p,q),true);
 const position=e.position,orientation=e.orientation;let changes=0;
 e.definitionChanged.addEventListener(()=>changes++);
 for(let i=0;i<120;i++)assert.equal(updateReplayPose(e,C.Cartesian3.clone(p),C.Quaternion.clone(q)),false);
 assert.equal(e.position,position);assert.equal(e.orientation,orientation);assert.equal(changes,0);
 const turned=C.Quaternion.fromAxisAngle(C.Cartesian3.UNIT_Z,.2);
 assert.equal(updateReplayPose(e,p,turned),true);assert.equal(e.position,position);assert.equal(changes,1);
 assert.equal(e.position.isConstant,true);assert.equal(e.orientation.isConstant,true);
});

test('route colour changes once on a new manoeuvre, without dynamic callbacks',()=>{
 const e=line();const material=e.polyline.material;let changes=0;
 material.definitionChanged.addEventListener(()=>changes++);
 assert.equal(updateReplayColour(e,C.Color.clone(C.Color.CYAN)),false);
 assert.equal(updateReplayColour(e,C.Color.ORANGE),true);
 assert.equal(updateReplayColour(e,C.Color.clone(C.Color.ORANGE)),false);
 assert.equal(changes,1);assert.equal(material.isConstant,true);
});
