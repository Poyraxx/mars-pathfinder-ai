import {useEffect,useRef,useState} from 'react';
import * as C from 'cesium';
import Lerc from 'lerc';
import 'cesium/Build/Cesium/Widgets/widgets.css';
import {useMission} from './store';
import {updateReplayLine,updateReplayPose,updateReplayColour,measuredReplayLine,keepReplayDatum} from './replayScene';

// Mars 2000 sphere used by the open MOLA/HRSC service and NASA Trek.
export const MARS=new C.Ellipsoid(3396190,3396190,3396190);
export const scheme=new C.GeographicTilingScheme({ellipsoid:MARS});
const terrainTiles=new Map<string,Promise<ReturnType<typeof Lerc.decode>>>();
let localDEM:{values:Float32Array;meta:{north_m:number;west_m:number;cell_m:number;shape:number[]}}|null=null;
let localMinimum:number|null=null;
export const localDEMMinimum=()=>localMinimum;
export const ROVER_MODEL='https://assets.science.nasa.gov/content/dam/science/psd/mars/resources/gltf_files/24584_Curiosity_static.glb';
let localDEMPromise:Promise<void>|null=null;
export function ensureLocalDEM(){
 if(!localDEMPromise)localDEMPromise=fetch('/api/regions/gale').then(r=>r.ok?r.json():null).then(async meta=>{if(!meta)return;const response=await fetch('/api/regions/gale/heightmap');if(!response.ok)return;localDEM={values:new Float32Array(await response.arrayBuffer()),meta};let low=Infinity;for(const h of localDEM.values)if(Number.isFinite(h))low=Math.min(low,h);localMinimum=Number.isFinite(low)?low:null;}).catch(()=>{});
 return localDEMPromise;
}
export function demHeightAt(lat:number,lon:number):number|null{
 if(!localDEM)return null;
 const m=localDEM.meta,gx=(3396190*C.Math.toRadians(lon)-m.west_m)/m.cell_m-.5,gy=(m.north_m-3396190*C.Math.toRadians(lat))/m.cell_m-.5;
 const w=m.shape[1],h=m.shape[0];if(gx<0||gy<0||gx>=w-1||gy>=h-1)return null;
 const i=Math.floor(gx),j=Math.floor(gy),ax=gx-i,ay=gy-j,v=localDEM.values;
 return (v[j*w+i]*(1-ax)+v[j*w+i+1]*ax)*(1-ay)+(v[(j+1)*w+i]*(1-ax)+v[(j+1)*w+i+1]*ax)*ay;
}

/** Replay annotations on the actual 1 m HiRISE DEM, independent of streamed
 * terrain LOD/classification rebuilds. The 15 cm lift is display separation.
 * Outside measured DEM coverage we retain Cesium's ground-clamped fallback. */
export function replayGroundLine(points:{lat:number;lon:number}[]){
 return measuredReplayLine(points,demHeightAt,MARS);
}
/** Display version of the greyscale HiRISE ortho: luminance mapped onto the surrounding Viking colour and edges
 * feathered, so the 25 cm patch blends into the 232 m colour base instead of sitting on it as a grey square.
 * Brightness (the measured signal) is preserved; the hue is a display tint, not HiRISE colour data. */
const tinted=new Map<string,Promise<string>>();
export function martianOrtho(url:string):Promise<string>{
 if(!tinted.has(url))tinted.set(url,new Promise((resolve,reject)=>{
  const img=new Image();img.crossOrigin='anonymous';
  img.onload=()=>{
   const w=img.naturalWidth,h=img.naturalHeight,cv=document.createElement('canvas');cv.width=w;cv.height=h;
   const ctx=cv.getContext('2d')!;ctx.drawImage(img,0,0);const d=ctx.getImageData(0,0,w,h),px=d.data;
   const dark=[52,36,32],mid=[118,86,78],light=[214,178,156],feather=Math.round(Math.min(w,h)*.025);
   for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    const i=(y*w+x)*4,l=Math.min(1,Math.max(0,(px[i]-82)/(170-82)));
    const a=l<.5?dark:mid,b=l<.5?mid:light,t=l<.5?l*2:(l-.5)*2;
    const e=Math.min(x,y,w-1-x,h-1-y)/feather,f=e>=1?1:e*e*(3-2*e);
    px[i]=a[0]+(b[0]-a[0])*t;px[i+1]=a[1]+(b[1]-a[1])*t;px[i+2]=a[2]+(b[2]-a[2])*t;px[i+3]=255*f;
   }
   ctx.putImageData(d,0,0);resolve(cv.toDataURL('image/png'));
  };
  img.onerror=()=>reject(new Error('ortho unavailable'));img.src=url;
 }));
 return tinted.get(url)!;
}

export function terrain(){
 return new C.CustomHeightmapTerrainProvider({width:65,height:65,tilingScheme:scheme,
  credit:'NASA / USGS / ESA / DLR / FU Berlin · MOLA/HRSC DEM · Esri hosting',
  callback:async(x,y,level)=>{
   const z=Math.min(level,7),factor=2**(level-z),tx=Math.floor(x/factor),ty=Math.floor(y/factor),key=`${z}/${ty}/${tx}`;
   if(!terrainTiles.has(key)){
    const promise=fetch(`/api/terrain/${key}`).then(r=>{if(!r.ok)throw new Error('DEM tile unavailable');return r.arrayBuffer();}).then(b=>Lerc.decode(b));
    terrainTiles.set(key,promise); promise.catch(()=>terrainTiles.delete(key));
    if(terrainTiles.size>80)terrainTiles.delete(terrainTiles.keys().next().value!);
   }
   const tile=await terrainTiles.get(key)!,out=new Float32Array(65*65),pixels=tile.pixels[0];
   for(let row=0;row<65;row++)for(let col=0;col<65;col++){
    const px=((x%factor)+col/64)/factor*(tile.width-1),py=((y%factor)+row/64)/factor*(tile.height-1);
    const ix=Math.floor(px),iy=Math.floor(py),jx=Math.min(ix+1,tile.width-1),jy=Math.min(iy+1,tile.height-1),a=px-ix,b=py-iy;
    const v=(pixels[iy*tile.width+ix]*(1-a)+pixels[iy*tile.width+jx]*a)*(1-b)+(pixels[jy*tile.width+ix]*(1-a)+pixels[jy*tile.width+jx]*a)*b;
    // Invalid source pixels must not create astronomical spikes.
    out[row*65+col]=Number.isFinite(v)&&v>-20000&&v<30000?v:0;
    if(localDEM&&level>=10){
     const rectangle=scheme.tileXYToRectangle(x,y,level),lon=C.Math.lerp(rectangle.west,rectangle.east,col/64),lat=C.Math.lerp(rectangle.north,rectangle.south,row/64);
     const gx=(3396190*lon-localDEM.meta.west_m)/localDEM.meta.cell_m-.5,gy=(localDEM.meta.north_m-3396190*lat)/localDEM.meta.cell_m-.5;
     const width=localDEM.meta.shape[1],height=localDEM.meta.shape[0];
     if(gx>=0&&gy>=0&&gx<width-1&&gy<height-1){
      const i=Math.floor(gx),j=Math.floor(gy),ax=gx-i,ay=gy-j,values=localDEM.values;
      const elevation=(values[j*width+i]*(1-ax)+values[j*width+i+1]*ax)*(1-ay)+(values[(j+1)*width+i]*(1-ax)+values[(j+1)*width+i+1]*ax)*ay;
      if(Number.isFinite(elevation))out[row*65+col]=elevation;
     }
    }
   }
   return out;
  }});
}
let flownMission='';
const typeColor:Record<string,string>={Mons:'#f0c58d',Tholus:'#f0c58d',Patera:'#f0c58d',Vallis:'#8ec9d6',Valles:'#8ec9d6',Chasma:'#8ec9d6',Planitia:'#b8c4a0',Planum:'#b8c4a0',Crater:'#d3c4b1'};
/** Oblique camera looking north at a point, so exaggerated relief reads as height. Altitude is clamped to a regional scale. */
function obliqueOver(v:C.Viewer,lat:number,lon:number,height:number){
 const alt=Math.min(1600000,Math.max(80000,height));
 v.camera.flyTo({destination:C.Cartesian3.fromRadians(lon,lat-alt*1.05/3396190,alt*1.15,MARS),orientation:{heading:0,pitch:C.Math.toRadians(-45),roll:0},duration:1.4});
}

const RELIEF_X=12;
/** Route heights for the exaggerated relief view: the route's own MOLA/HRSC elevation profile, scaled like the terrain and lifted slightly so it is never buried. */
function reliefHeights(route:{points:{lat:number;lon:number}[];profile?:{distance_m:number;elevation_m:number}[]}):number[]|null{
 const prof=route.profile;if(!prof?.length)return null;
 const R=3396190,rad=Math.PI/180;let d=0;const out:number[]=[];
 route.points.forEach((p,i)=>{
  if(i){const q=route.points[i-1],a=Math.sin((p.lat-q.lat)*rad/2)**2+Math.cos(p.lat*rad)*Math.cos(q.lat*rad)*Math.sin((p.lon-q.lon)*rad/2)**2;d+=2*R*Math.asin(Math.min(1,Math.sqrt(a)));}
  out.push(d);
 });
 const scale=prof[prof.length-1].distance_m/(d||1);
 return out.map(x=>{x*=scale;let j=0;while(j<prof.length-2&&prof[j+1].distance_m<x)j++;const a=prof[j],b=prof[Math.min(j+1,prof.length-1)],t=b.distance_m>a.distance_m?Math.min(1,Math.max(0,(x-a.distance_m)/(b.distance_m-a.distance_m))):0;return (a.elevation_m+(b.elevation_m-a.elevation_m)*t)*RELIEF_X+1800;});
}

export default function Globe(){
 const mount=useRef<HTMLDivElement>(null),viewerRef=useRef<C.Viewer|null>(null),layerRef=useRef<C.ImageryLayer|null>(null),riskRef=useRef<C.ImageryLayer|null>(null),orthoRef=useRef<C.ImageryLayer|null>(null),tilesetRef=useRef<C.Cesium3DTileset|null>(null);
 const [status,setStatus]=useState('CONNECTING TO MARS'),[renderer,setRenderer]=useState('MOLA / HRSC · OPEN TERRAIN');
 const [groundReady,setGroundReady]=useState(false);
 const {catalog,region,view,layer,showRisk,start,goal,stops,mission,activeRoute,pick,exploreTarget,orbital,mode,replay,replayStep,relief,overlay}=useMission();
 const overlayRef=useRef<C.ImageryLayer|null>(null);
 useEffect(()=>{
  if(!mount.current)return;
  C.Ellipsoid.default=MARS;
  const viewer=new C.Viewer(mount.current,{animation:false,timeline:false,baseLayerPicker:false,geocoder:false,sceneModePicker:false,homeButton:false,fullscreenButton:false,navigationHelpButton:false,selectionIndicator:false,infoBox:false,baseLayer:false,
   terrainProvider:terrain(),globe:new C.Globe(MARS),skyAtmosphere:false,requestRenderMode:true,maximumRenderTimeChange:Infinity,contextOptions:{webgl:{alpha:true}}});
  viewerRef.current=viewer;if(import.meta.env.DEV)(window as unknown as {__viewer:unknown}).__viewer=viewer;
  void ensureLocalDEM().then(()=>{if(!viewer.isDestroyed())setGroundReady(true);});
  viewer.scene.backgroundColor=C.Color.fromCssColorString('#070b10');
  viewer.scene.globe.baseColor=C.Color.fromCssColorString('#9a5940');
  viewer.scene.globe.enableLighting=false;
  viewer.scene.globe.maximumScreenSpaceError=3;
  viewer.scene.globe.tileCacheSize=80;
  viewer.scene.globe.depthTestAgainstTerrain=true;
  viewer.scene.screenSpaceCameraController.minimumZoomDistance=5;
  viewer.scene.screenSpaceCameraController.maximumZoomDistance=19000000;
  viewer.camera.setView({destination:C.Cartesian3.fromDegrees(42,17,10500000,MARS)});
  const names=viewer.scene.primitives.add(new C.LabelCollection({scene:viewer.scene}));
  fetch('/api/nomenclature').then(r=>r.ok?r.json():null).then(catalogue=>{
   if(!catalogue||viewer.isDestroyed())return;
   useMission.getState().set({features:catalogue.features});
   for(const f of catalogue.features as {name:string;type:string;diameter_km:number;lat:number;lon:number}[]){
    // Larger features stay visible from further away; tiny craters appear only close up.
    const far=Math.min(17000000,Math.max(120000,f.diameter_km*14000));
    names.add({position:C.Cartesian3.fromDegrees(f.lon,f.lat,f.type==='Mons'||f.type==='Tholus'?22000:3000,MARS),text:f.diameter_km>=300?f.name.toUpperCase():f.name,
     font:f.diameter_km>=300?'600 11px sans-serif':'10px sans-serif',fillColor:C.Color.fromCssColorString(typeColor[f.type]||'#cdbfae'),outlineColor:C.Color.fromCssColorString('#0b1015'),outlineWidth:3,style:C.LabelStyle.FILL_AND_OUTLINE,
     distanceDisplayCondition:new C.DistanceDisplayCondition(0,far),scaleByDistance:new C.NearFarScalar(50000,1.05,15000000,.8)});
   }
   viewer.scene.requestRender();
  }).catch(()=>{});
  const load=viewer.scene.globe.tileLoadProgressEvent.addEventListener(n=>setStatus(n?'STREAMING PLANETARY DATA':'PLANETARY DATA CONNECTED'));
  const error=viewer.terrainProvider.errorEvent.addEventListener(()=>setStatus('TERRAIN SERVICE RETRYING'));
  const handler=new C.ScreenSpaceEventHandler(viewer.scene.canvas);
  handler.setInputAction((event:{position:C.Cartesian2})=>{
   if(!useMission.getState().pick)return;
   const ray=viewer.camera.getPickRay(event.position); if(!ray)return;
   const world=tilesetRef.current?viewer.scene.pickPosition(event.position):viewer.scene.globe.pick(ray,viewer.scene);
   if(!world)return;
   const cart=MARS.cartesianToCartographic(world);
   useMission.getState().selectPoint({lat:C.Math.toDegrees(cart.latitude),lon:((C.Math.toDegrees(cart.longitude)+180)%360+360)%360-180});
  },C.ScreenSpaceEventType.LEFT_CLICK);
  const token=import.meta.env.VITE_CESIUM_ION_TOKEN;
  if(token){C.Ion.defaultAccessToken=token;C.Cesium3DTileset.fromIonAssetId(3644333,{ellipsoid:MARS}).then(tileset=>{
   if(viewer.isDestroyed())return;viewer.scene.primitives.add(tileset);tilesetRef.current=tileset;viewer.scene.globe.show=false;setRenderer('CESIUM MARS · GLOBAL 3D TILES');
  }).catch(()=>setRenderer('MOLA / HRSC · ION UNAVAILABLE'));}
  return()=>{load();error();handler.destroy();viewer.destroy();viewerRef.current=null;};
 },[]);
 useEffect(()=>{
  const v=viewerRef.current;if(!v)return;
  // Keep negative-elevation replay annotations inside the horizon bound during
  // streamed LOD transitions. Terrain depth and on-demand rendering stay intact.
  if(mode==='replay')return keepReplayDatum(v.scene,localDEMMinimum);
 },[mode]);
 useEffect(()=>{
  // Environment / geology overlays on top of the base map (one at a time, toggled like the relief view).
  const v=viewerRef.current;if(!v)return;
  if(overlayRef.current){v.imageryLayers.remove(overlayRef.current);overlayRef.current=null;}
  for(const e of [...v.entities.values])if(String(e.id).startsWith('conf-'))v.entities.remove(e);
  const meta=catalog?.overlays?.find(o=>o.id===overlay);
  if(overlay==='geology'){
   C.SingleTileImageryProvider.fromUrl('/api/geology/overlay.png',{rectangle:C.Rectangle.fromDegrees(-180,-90,180,90),ellipsoid:MARS,credit:'USGS SIM 3292 (Tanaka et al. 2014)'}).then(p=>{if(!v.isDestroyed()&&useMission.getState().overlay==='geology'){overlayRef.current=v.imageryLayers.addImageryProvider(p);overlayRef.current.alpha=.62;v.scene.requestRender();}});
  }else if(overlay==='confidence'){
   for(const r of catalog?.regions??[]){
    const [w,so,e,n]=r.bounds;v.entities.add({id:`conf-dem-${r.id}`,rectangle:{coordinates:C.Rectangle.fromDegrees(w,so,e,n),material:C.Color.fromCssColorString('#5fd48a').withAlpha(.22),outline:true,outlineColor:C.Color.fromCssColorString('#5fd48a'),height:0,classificationType:C.ClassificationType.BOTH},
     position:C.Cartesian3.fromDegrees((w+e)/2,n,0,MARS),label:{text:'Measured: HiRISE 1 m DEM',font:'600 11px sans-serif',fillColor:C.Color.fromCssColorString('#7ff0a8'),pixelOffset:new C.Cartesian2(0,-12),heightReference:C.HeightReference.CLAMP_TO_GROUND,disableDepthTestDistance:Infinity,distanceDisplayCondition:new C.DistanceDisplayCondition(0,400000)}});
    if(r.ortho_bounds){const [a,b,c,d]=r.ortho_bounds;v.entities.add({id:`conf-ortho-${r.id}`,rectangle:{coordinates:C.Rectangle.fromDegrees(a,b,c,d),material:C.Color.fromCssColorString('#5fd48a').withAlpha(.12),outline:true,outlineColor:C.Color.fromCssColorString('#b8ffd0'),height:0,classificationType:C.ClassificationType.BOTH}});}
   }
   for(const f of (catalog?.frames??[]) as unknown as {id:string;registration?:{verified?:boolean};places?:{row?:{planetocentric_latitude?:string;longitude?:string}}}[]){
    const lat=Number(f.places?.row?.planetocentric_latitude),lon=Number(f.places?.row?.longitude);if(!Number.isFinite(lat)||!Number.isFinite(lon))continue;
    const ok=!!f.registration?.verified;v.entities.add({id:`conf-frame-${f.id}`,position:C.Cartesian3.fromDegrees(lon,lat,0,MARS),point:{pixelSize:ok?8:6,color:C.Color.fromCssColorString(ok?'#5fd48a':'#f2a65a'),outlineColor:C.Color.fromCssColorString('#0b1015'),outlineWidth:1.5,heightReference:C.HeightReference.CLAMP_TO_GROUND,disableDepthTestDistance:Infinity}});
   }
  }else if(meta){
   const provider=new C.UrlTemplateImageryProvider({url:`/api/overlays/${overlay}/{z}/{y}/{x}`,tilingScheme:scheme,maximumLevel:meta.max_level??4,credit:meta.source});
   overlayRef.current=v.imageryLayers.addImageryProvider(provider);overlayRef.current.alpha=.72;
  }
  v.scene.requestRender();
 },[overlay,catalog]);
 useEffect(()=>{
  const viewer=viewerRef.current;if(!viewer)return;
  if(layerRef.current){viewer.imageryLayers.remove(layerRef.current);layerRef.current=null;}
  const meta=useMission.getState().catalog?.layers.find(l=>l.id===layer);
  const provider=new C.UrlTemplateImageryProvider({url:`/api/tiles/${layer}/{z}/{y}/{x}`,tilingScheme:scheme,maximumLevel:layer==='themis'?9:7,credit:meta?.mission||'NASA / USGS · Mars Trek',rectangle:layer==='themis'?C.Rectangle.fromDegrees(-180,-65,180,65):C.Rectangle.MAX_VALUE});
  layerRef.current=viewer.imageryLayers.addImageryProvider(provider,0);
  provider.errorEvent.addEventListener(()=>setStatus('IMAGERY SERVICE RETRYING'));
  viewer.scene.requestRender();
 },[layer]);
 useEffect(()=>{
  const v=viewerRef.current;if(!v)return;
  if(view==='orbital')v.camera.flyTo({destination:C.Cartesian3.fromDegrees(exploreTarget?.lon??region?.center.lon??42,exploreTarget?.lat??region?.center.lat??17,exploreTarget?.alt??10000000,MARS),duration:1.8});
  else if(region)v.camera.flyTo({destination:C.Cartesian3.fromDegrees(region.center.lon,region.center.lat,(region.display_surface_height_m??-4517)+(view==='terrain'?850:1200),MARS),orientation:{heading:0,pitch:C.Math.toRadians(-80),roll:0},duration:2});
 },[region,view,exploreTarget]);
 useEffect(()=>{
  const v=viewerRef.current;if(!v)return;
  for(const e of [...v.entities.values])if(!String(e.id).startsWith('orbital-')&&!String(e.id).startsWith('replay-')&&!String(e.id).startsWith('conf-'))v.entities.remove(e);
  const coverage=region?.bounds??mission?.region?.bounds;
  if(coverage){const [w,s,e,n]=coverage;v.entities.add({id:'coverage',rectangle:{coordinates:C.Rectangle.fromDegrees(w,s,e,n),material:C.Color.fromCssColorString('#6faeab').withAlpha(.045),outline:true,outlineColor:C.Color.fromCssColorString('#64b9b0'),height:0,classificationType:C.ClassificationType.BOTH}});}
  for(const [id,p,color]of [['start',start,'#75dbc1'],['goal',goal,'#f4b06f'],...stops.map((p,i)=>[`stop-${i}`,p,'#b7b4ed'])] as [string,{lat:number;lon:number}|null,string][]){if(p)v.entities.add({id,position:C.Cartesian3.fromDegrees(p.lon,p.lat,0,MARS),point:{pixelSize:10,color:C.Color.fromCssColorString(color),outlineColor:C.Color.fromCssColorString('#111b22'),outlineWidth:3,heightReference:C.HeightReference.CLAMP_TO_GROUND,disableDepthTestDistance:Infinity},label:{text:id==='start'?'START':id==='goal'?'DESTINATION':`SCIENCE ${id.split('-')[1]}`,font:'10px sans-serif',pixelOffset:new C.Cartesian2(0,-23),fillColor:C.Color.fromCssColorString(color),heightReference:C.HeightReference.CLAMP_TO_GROUND,disableDepthTestDistance:Infinity}});}
  const route=mission?.routes.find(r=>r.mode===activeRoute&&r.status==='ready');
  if(route?.points.length){
   const lifted=relief&&mode!=='replay'?reliefHeights(route):null;
   if(lifted)v.entities.add({id:'active-route',polyline:{positions:C.Cartesian3.fromDegreesArrayHeights(route.points.flatMap((p,i)=>[p.lon,p.lat,lifted[i]]),MARS),width:10,material:new C.PolylineOutlineMaterialProperty({color:C.Color.fromCssColorString('#3ce1ff'),outlineColor:C.Color.fromCssColorString('#06121a'),outlineWidth:2.5}),depthFailMaterial:new C.ColorMaterialProperty(C.Color.fromCssColorString('#3ce1ff').withAlpha(.45))}});
   else v.entities.add({id:'active-route',polyline:{positions:C.Cartesian3.fromDegreesArray(route.points.flatMap(p=>[p.lon,p.lat]),MARS),width:4,material:new C.PolylineGlowMaterialProperty({glowPower:.16,color:C.Color.fromCssColorString('#75dbc1')}),clampToGround:true}});
  }
  const old=mission?.previous_route;
  if(old?.points.length)v.entities.add({id:'old-route',polyline:{positions:C.Cartesian3.fromDegreesArray(old.points.flatMap(p=>[p.lon,p.lat]),MARS),width:2,material:new C.PolylineDashMaterialProperty({color:C.Color.fromCssColorString('#e7a66b').withAlpha(.6)}),clampToGround:true}});
  const back=mission?.eva?.status==='ready'?mission.eva.return?.points:undefined;
  if(back&&back.length>1)v.entities.add({id:'return-route',polyline:{positions:C.Cartesian3.fromDegreesArray(back.flatMap(p=>[p.lon,p.lat]),MARS),width:3,material:new C.PolylineDashMaterialProperty({color:C.Color.fromCssColorString('#f2a65a'),dashLength:14}),clampToGround:true}});
  if(!region&&route?.points.length&&mission?.region?.bounds&&flownMission!==mission.id){flownMission=mission.id;const [w,s,e,n]=mission.region.bounds;if(relief&&mode!=='replay'){const la=route.points.map(p=>p.lat),lo=route.points.map(p=>p.lon),s2=Math.min(...la),n2=Math.max(...la),w2=Math.min(...lo),e2=Math.max(...lo);obliqueOver(v,C.Math.toRadians((s2+n2)/2),C.Math.toRadians((w2+e2)/2),Math.max(e2-w2,n2-s2)*59000*.95);}else v.camera.flyTo({destination:C.Rectangle.fromDegrees(w,s,e,n),duration:1.6});}
  v.scene.requestRender();
 },[region,start,goal,stops,mission,activeRoute,relief,mode]);
 useEffect(()=>{
  const v=viewerRef.current;if(!v)return;
  if(orthoRef.current){v.imageryLayers.remove(orthoRef.current);orthoRef.current=null;}
  const oregion=region??(mode==='replay'?catalog?.regions[0]??null:null);
  if(oregion?.ortho_bounds&&layer==='viking'){
   const [w,s,e,n]=oregion.ortho_bounds;
   martianOrtho(`/api/regions/${oregion.id}/ortho.png`).then(url=>C.SingleTileImageryProvider.fromUrl(url,{rectangle:C.Rectangle.fromDegrees(w,s,e,n),ellipsoid:MARS,credit:'NASA/JPL-Caltech/USGS/UArizona · HiRISE PLACES 25cm ortho (greyscale, tinted for display)'})).then(provider=>{if(!v.isDestroyed()&&!orthoRef.current){orthoRef.current=v.imageryLayers.addImageryProvider(provider);v.scene.requestRender();}}).catch(()=>setStatus('REGIONAL IMAGERY UNAVAILABLE'));
  }
 },[region,layer,mode,catalog?.regions]);
 useEffect(()=>{
  const v=viewerRef.current;if(!v)return;
  if(riskRef.current){v.imageryLayers.remove(riskRef.current);riskRef.current=null;}
  if(showRisk&&region){const [w,s,e,n]=region.bounds;C.SingleTileImageryProvider.fromUrl(`/api/regions/${region.id}/risk.png`,{rectangle:C.Rectangle.fromDegrees(w,s,e,n),ellipsoid:MARS}).then(provider=>{if(!v.isDestroyed()){riskRef.current=v.imageryLayers.addImageryProvider(provider);v.scene.requestRender();}});}
 },[showRisk,region]);
 useEffect(()=>{
  const v=viewerRef.current;if(!v)return;
  for(const e of [...v.entities.values])if(String(e.id).startsWith('orbital-'))v.entities.remove(e);
  if(!orbital||orbital.mission_id!==mission?.id){v.scene.requestRender();return;}
  const colours:Record<string,string>={crater:'#f0c58d',dark_dune:'#785cc8',slope_streak:'#5ac8e6',bright_dune:'#fae696',impact_ejecta:'#ee785a',swiss_cheese:'#96dca0',spider:'#dc78c8',other:'#9aa9ad'};
  const half=orbital.window_m/2/3396190*180/Math.PI;
  orbital.windows.forEach((w,i)=>{const dl=half/Math.max(Math.cos(w.lat*Math.PI/180),.05);v.entities.add({id:`orbital-${i}`,rectangle:{coordinates:C.Rectangle.fromDegrees(w.lon-dl,w.lat-half,w.lon+dl,w.lat+half),material:C.Color.fromCssColorString(colours[w.class]??'#9aa9ad').withAlpha(w.class==='other'?.06:.28),outline:true,outlineColor:C.Color.fromCssColorString(colours[w.class]??'#9aa9ad').withAlpha(.8),height:0,classificationType:C.ClassificationType.BOTH},
   ...(w.class!=='other'?{position:C.Cartesian3.fromDegrees(w.lon,w.lat,0,MARS),label:{text:`${w.class.replaceAll('_',' ')} ${(w.score*100).toFixed(0)}%`,font:'10px sans-serif',fillColor:C.Color.fromCssColorString(colours[w.class]),pixelOffset:new C.Cartesian2(0,-12),heightReference:C.HeightReference.CLAMP_TO_GROUND,disableDepthTestDistance:Infinity}}:{})});});
  v.scene.requestRender();
 },[orbital,mission?.id]);
 // Keep replay entities and constant geometry alive across renders and zooms.
 // Dynamic clamped callbacks rebuilt primitives even while the replay was paused.
 useEffect(()=>{
  const v=viewerRef.current;if(!v)return;
  for(const e of [...v.entities.values])if(String(e.id).startsWith('replay-'))v.entities.remove(e);
  if(mode!=='replay'||!replay){v.scene.requestRender();return;}
  // Actual driven path: every logged PLACES drive step between acquisitions (forward, reverse, turns),
  // falling back to the straight pose-to-pose line where no drive log exists.
  const driven=replay.steps.map((st,i)=>st.driven?.length?st.driven.map(([lat,lon])=>({lat,lon})):i?[replay.trail[i-1],replay.trail[i]]:[replay.trail[0]]);
  const upTo=(k:number)=>driven.slice(0,k+1).flat();
  v.entities.add({id:'replay-trail-all',polyline:{...replayGroundLine(upTo(driven.length-1)),width:2,material:new C.PolylineDashMaterialProperty({color:C.Color.WHITE.withAlpha(.35),dashLength:8})}});
  v.entities.add({id:'replay-trail',polyline:{positions:[],width:4,material:new C.PolylineOutlineMaterialProperty({color:C.Color.fromCssColorString('#ff4a3d'),outlineColor:C.Color.fromCssColorString('#2a0805'),outlineWidth:1}),clampToGround:true}});
  v.entities.add({id:'replay-route',polyline:{positions:[],width:6,material:new C.PolylineGlowMaterialProperty({glowPower:.22,color:C.Color.fromCssColorString('#3ce1ff')}),clampToGround:true}});
  const goalHeight=demHeightAt(replay.goal.lat,replay.goal.lon);
  v.entities.add({id:'replay-goal',position:C.Cartesian3.fromDegrees(replay.goal.lon,replay.goal.lat,(goalHeight??0)+.15,MARS),point:{pixelSize:12,color:C.Color.fromCssColorString('#f2a65a'),outlineColor:C.Color.fromCssColorString('#0b1015'),outlineWidth:3,heightReference:goalHeight===null?C.HeightReference.CLAMP_TO_GROUND:C.HeightReference.NONE,disableDepthTestDistance:Infinity},label:{text:'DESTINATION',font:'600 11px sans-serif',fillColor:C.Color.fromCssColorString('#f2a65a'),pixelOffset:new C.Cartesian2(0,-20),heightReference:goalHeight===null?C.HeightReference.CLAMP_TO_GROUND:C.HeightReference.NONE,disableDepthTestDistance:Infinity}});
  const hasDEM=demHeightAt(replay.steps[0].rover.lat,replay.steps[0].rover.lon)!==null;
  v.entities.add({id:'replay-rover',
   model:{uri:ROVER_MODEL,minimumPixelSize:56,maximumScale:400,heightReference:hasDEM?C.HeightReference.NONE:C.HeightReference.CLAMP_TO_GROUND},
   point:{pixelSize:9,color:C.Color.WHITE,outlineColor:C.Color.fromCssColorString('#3ce1ff'),outlineWidth:3,heightReference:hasDEM?C.HeightReference.NONE:C.HeightReference.CLAMP_TO_GROUND,disableDepthTestDistance:Infinity}});
  v.scene.requestRender();
 },[mode,replay,groundReady]);
 useEffect(()=>{
  const v=viewerRef.current;if(!v||mode!=='replay'||!replay)return;
  const st=replay.steps[Math.min(replayStep,replay.steps.length-1)];
  const done=replay.steps.slice(0,replayStep+1).flatMap((s,i)=>s.driven?.length?s.driven.map(([lat,lon])=>({lat,lon})):i?[replay.trail[i-1],replay.trail[i]]:[replay.trail[0]]);
  const trail=replayGroundLine(done.length>1?done:[done[0],done[0]]),route=replayGroundLine([st.rover,...(st.route.length?st.route:[st.rover])]);
  updateReplayLine(v.entities.getById('replay-trail')!,trail.positions,trail.clampToGround);
  updateReplayLine(v.entities.getById('replay-route')!,route.positions,route.clampToGround);
  updateReplayColour(v.entities.getById('replay-route')!,C.Color.fromCssColorString(st.event==='u_turn'?'#f2a65a':'#3ce1ff'));
  const position=C.Cartesian3.fromDegrees(st.rover.lon,st.rover.lat,(demHeightAt(st.rover.lat,st.rover.lon)??0)+.05,MARS);
  const poseChanged=updateReplayPose(v.entities.getById('replay-rover')!,position,C.Transforms.headingPitchRollQuaternion(position,new C.HeadingPitchRoll(C.Math.toRadians(st.rover.yaw_deg-90),0,0),MARS));
  if(useMission.getState().followRover&&poseChanged){
   // Follow mode: keep the current zoom, re-centre on the rover as the replay advances.
   const st=replay.steps[Math.min(replayStep,replay.steps.length-1)];
   const ground=v.scene.globe.getHeight(C.Cartographic.fromDegrees(st.rover.lon,st.rover.lat))??catalog?.regions[0]?.display_surface_height_m??-4517;
   const above=Math.max(15,Math.min(400,v.camera.positionCartographic.height-ground));
   v.camera.setView({destination:C.Cartesian3.fromDegrees(st.rover.lon,st.rover.lat,ground+above,MARS),orientation:{heading:v.camera.heading,pitch:v.camera.pitch,roll:0}});
  }
  v.scene.requestRender();
 },[mode,replay,replayStep,groundReady]);
 useEffect(()=>{
  const v=viewerRef.current;if(!v||mode!=='replay'||!replay)return;
  const fit=()=>{
   const positions=[...replay.trail,replay.goal,...replay.steps.flatMap(s=>s.route)];
   const lats=positions.map(p=>p.lat),lons=positions.map(p=>p.lon);
   const clat=(Math.min(...lats)+Math.max(...lats))/2,clon=(Math.min(...lons)+Math.max(...lons))/2;
   const span=Math.max((Math.max(...lats)-Math.min(...lats))*Math.PI/180*3396190,(Math.max(...lons)-Math.min(...lons))*Math.PI/180*3396190,18)*1.35+12;
   const aspect=v.canvas.clientWidth/Math.max(1,v.canvas.clientHeight);
   const height=span/(2*Math.tan(Math.PI/6)*Math.min(1,aspect));
   const ground=catalog?.regions[0]?.display_surface_height_m??-4517;
   v.camera.flyTo({destination:C.Cartesian3.fromDegrees(clon,clat,ground+height,MARS),orientation:{heading:0,pitch:-Math.PI/2,roll:0},duration:1.2});
  };
  const close=()=>{
   const st=replay.steps[Math.min(useMission.getState().replayStep,replay.steps.length-1)];
   const ground=v.scene.globe.getHeight(C.Cartographic.fromDegrees(st.rover.lon,st.rover.lat))??catalog?.regions[0]?.display_surface_height_m??-4517;
   v.camera.flyTo({destination:C.Cartesian3.fromDegrees(st.rover.lon,st.rover.lat,ground+38,MARS),orientation:{heading:0,pitch:-Math.PI/2,roll:0},duration:1});
  };
  fit();window.addEventListener('replay-fit',fit);window.addEventListener('replay-rover',close);
  return()=>{window.removeEventListener('replay-fit',fit);window.removeEventListener('replay-rover',close);};
 },[mode,replay,catalog?.regions]);
 useEffect(()=>{
  const v=viewerRef.current;if(!v)return;
  // Relief view: exaggerate real MOLA/HRSC heights so mountains rise and basins sink.
  (v.scene as unknown as {verticalExaggeration:number}).verticalExaggeration=relief&&mode!=='replay'?RELIEF_X:1;
  if(relief&&mode!=='replay'){
   // Tilt the view so the exaggerated relief reads as height, not just colour.
   const canvas=v.scene.canvas,centre=v.camera.pickEllipsoid(new C.Cartesian2(canvas.clientWidth/2,canvas.clientHeight/2),MARS);
   if(centre){
    const c=MARS.cartesianToCartographic(centre);
    obliqueOver(v,c.latitude,c.longitude,MARS.cartesianToCartographic(v.camera.position).height);
   }
  }
  v.scene.requestRender();
 },[relief,mode]);
 return <div className={`globe-wrap ${pick?'picking':''}`}><div className="globe-canvas" ref={mount}/><div className="globe-vignette"/><div className="globe-source"><span className="status-dot"/>{status}<span className="source-divider">/</span>{renderer}</div>{pick&&<div className="pick-prompt">Select {pick==='stop'?'a science waypoint':`the ${pick} point`} on the terrain <button onClick={()=>useMission.getState().set({pick:null})}>Cancel</button></div>}</div>;
}
