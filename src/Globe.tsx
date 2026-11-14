import {useEffect,useRef,useState} from 'react';
import * as C from 'cesium';
import Lerc from 'lerc';
import 'cesium/Build/Cesium/Widgets/widgets.css';
import {useMission} from './store';

// Mars 2000 sphere used by the open MOLA/HRSC service and NASA Trek.
const MARS=new C.Ellipsoid(3396190,3396190,3396190);
const scheme=new C.GeographicTilingScheme({ellipsoid:MARS});
const terrainTiles=new Map<string,Promise<ReturnType<typeof Lerc.decode>>>();
let localDEM:{values:Float32Array;meta:{north_m:number;west_m:number;cell_m:number;shape:number[]}}|null=null;
function terrain(){
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
const landmarks=[{name:'OLYMPUS MONS',lat:18.65,lon:-133.8},{name:'VALLES MARINERIS',lat:-13.9,lon:-59.2},{name:'GALE CRATER',lat:-4.59,lon:137.44},{name:'JEZERO CRATER',lat:18.38,lon:77.58},{name:'HELLAS PLANITIA',lat:-42.4,lon:70.5},{name:'ELYSIUM PLANITIA',lat:4.5,lon:135.9}];
export default function Globe(){
 const mount=useRef<HTMLDivElement>(null),viewerRef=useRef<C.Viewer|null>(null),layerRef=useRef<C.ImageryLayer|null>(null),riskRef=useRef<C.ImageryLayer|null>(null),orthoRef=useRef<C.ImageryLayer|null>(null),tilesetRef=useRef<C.Cesium3DTileset|null>(null);
 const [status,setStatus]=useState('CONNECTING TO MARS'),[renderer,setRenderer]=useState('MOLA / HRSC · OPEN TERRAIN');
 const {region,view,layer,showRisk,start,goal,stops,mission,activeRoute,pick,exploreTarget}=useMission();
 useEffect(()=>{
  if(!mount.current)return;
  C.Ellipsoid.default=MARS;
  const viewer=new C.Viewer(mount.current,{animation:false,timeline:false,baseLayerPicker:false,geocoder:false,sceneModePicker:false,homeButton:false,fullscreenButton:false,navigationHelpButton:false,selectionIndicator:false,infoBox:false,baseLayer:false,
   terrainProvider:terrain(),globe:new C.Globe(MARS),skyAtmosphere:false,requestRenderMode:true,maximumRenderTimeChange:Infinity,contextOptions:{webgl:{alpha:true}}});
  viewerRef.current=viewer;
  fetch('/api/regions/gale').then(r=>r.ok?r.json():null).then(async meta=>{
   if(!meta)return;const response=await fetch('/api/regions/gale/heightmap');if(!response.ok)return;
   localDEM={values:new Float32Array(await response.arrayBuffer()),meta};if(!viewer.isDestroyed())viewer.scene.globe.tileCacheSize=80;
  }).catch(()=>{});
  viewer.scene.backgroundColor=C.Color.fromCssColorString('#070b10');
  viewer.scene.globe.baseColor=C.Color.fromCssColorString('#9a5940');
  viewer.scene.globe.enableLighting=false;
  viewer.scene.globe.maximumScreenSpaceError=3;
  viewer.scene.globe.tileCacheSize=80;
  viewer.scene.globe.depthTestAgainstTerrain=true;
  viewer.scene.screenSpaceCameraController.minimumZoomDistance=5;
  viewer.scene.screenSpaceCameraController.maximumZoomDistance=19000000;
  viewer.camera.setView({destination:C.Cartesian3.fromDegrees(42,17,10500000,MARS)});
  for(const p of landmarks)viewer.entities.add({id:`landmark-${p.name}`,position:C.Cartesian3.fromDegrees(p.lon,p.lat,100,MARS),label:{text:p.name,font:'10px sans-serif',fillColor:C.Color.fromCssColorString('#d3c4b1'),outlineWidth:2,style:C.LabelStyle.FILL_AND_OUTLINE,pixelOffset:new C.Cartesian2(0,-10),distanceDisplayCondition:new C.DistanceDisplayCondition(20000,16000000),scaleByDistance:new C.NearFarScalar(20000,1,15000000,.75),disableDepthTestDistance:Number.POSITIVE_INFINITY}});
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
  const viewer=viewerRef.current;if(!viewer)return;
  if(layerRef.current){viewer.imageryLayers.remove(layerRef.current);layerRef.current=null;}
  const meta=useMission.getState().catalog?.layers.find(l=>l.id===layer);
  const provider=new C.UrlTemplateImageryProvider({url:`/api/tiles/${layer}/{z}/{y}/{x}`,tilingScheme:scheme,maximumLevel:layer==='themis'?9:7,credit:meta?.mission||'NASA / USGS · Mars Trek',rectangle:layer==='themis'?C.Rectangle.fromDegrees(-180,-65,180,65):C.Rectangle.MAX_VALUE});
  layerRef.current=viewer.imageryLayers.addImageryProvider(provider);
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
  for(const entity of [...v.entities.values])if(!String(entity.id).startsWith('landmark-'))v.entities.remove(entity);
  if(region){const [w,s,e,n]=region.bounds;v.entities.add({id:'coverage',rectangle:{coordinates:C.Rectangle.fromDegrees(w,s,e,n),material:C.Color.fromCssColorString('#6faeab').withAlpha(.045),outline:true,outlineColor:C.Color.fromCssColorString('#64b9b0'),height:0,classificationType:C.ClassificationType.BOTH}});}
  for(const [id,p,color]of [['start',start,'#75dbc1'],['goal',goal,'#f4b06f'],...stops.map((p,i)=>[`stop-${i}`,p,'#b7b4ed'])] as [string,{lat:number;lon:number}|null,string][]){if(p)v.entities.add({id,position:C.Cartesian3.fromDegrees(p.lon,p.lat,0,MARS),point:{pixelSize:10,color:C.Color.fromCssColorString(color),outlineColor:C.Color.fromCssColorString('#111b22'),outlineWidth:3,heightReference:C.HeightReference.CLAMP_TO_GROUND,disableDepthTestDistance:Infinity},label:{text:id==='start'?'START':id==='goal'?'DESTINATION':`SCIENCE ${id.split('-')[1]}`,font:'10px sans-serif',pixelOffset:new C.Cartesian2(0,-23),fillColor:C.Color.fromCssColorString(color),heightReference:C.HeightReference.CLAMP_TO_GROUND,disableDepthTestDistance:Infinity}});}
  const route=mission?.routes.find(r=>r.mode===activeRoute&&r.status==='ready');
  if(route?.points.length)v.entities.add({id:'active-route',polyline:{positions:C.Cartesian3.fromDegreesArray(route.points.flatMap(p=>[p.lon,p.lat]),MARS),width:4,material:new C.PolylineGlowMaterialProperty({glowPower:.16,color:C.Color.fromCssColorString('#75dbc1')}),clampToGround:true}});
  const old=mission?.previous_route;
  if(old?.points.length)v.entities.add({id:'old-route',polyline:{positions:C.Cartesian3.fromDegreesArray(old.points.flatMap(p=>[p.lon,p.lat]),MARS),width:2,material:new C.PolylineDashMaterialProperty({color:C.Color.fromCssColorString('#e7a66b').withAlpha(.6)}),clampToGround:true}});
  v.scene.requestRender();
 },[region,start,goal,stops,mission,activeRoute]);
 useEffect(()=>{
  const v=viewerRef.current;if(!v)return;
  if(orthoRef.current){v.imageryLayers.remove(orthoRef.current);orthoRef.current=null;}
  if(region?.ortho_bounds&&layer==='viking'){
   const [w,s,e,n]=region.ortho_bounds;
   C.SingleTileImageryProvider.fromUrl(`/api/regions/${region.id}/ortho.png`,{rectangle:C.Rectangle.fromDegrees(w,s,e,n),ellipsoid:MARS,credit:'NASA/JPL-Caltech/USGS/UArizona · HiRISE PLACES 25cm ortho'}).then(provider=>{if(!v.isDestroyed()){orthoRef.current=v.imageryLayers.addImageryProvider(provider);v.scene.requestRender();}}).catch(()=>setStatus('REGIONAL IMAGERY UNAVAILABLE'));
  }
 },[region,layer]);
 useEffect(()=>{
  const v=viewerRef.current;if(!v)return;
  if(riskRef.current){v.imageryLayers.remove(riskRef.current);riskRef.current=null;}
  if(showRisk&&region){const [w,s,e,n]=region.bounds;C.SingleTileImageryProvider.fromUrl(`/api/regions/${region.id}/risk.png`,{rectangle:C.Rectangle.fromDegrees(w,s,e,n),ellipsoid:MARS}).then(provider=>{if(!v.isDestroyed()){riskRef.current=v.imageryLayers.addImageryProvider(provider);v.scene.requestRender();}});}
 },[showRisk,region]);
 return <div className={`globe-wrap ${pick?'picking':''}`}><div className="globe-canvas" ref={mount}/><div className="globe-vignette"/><div className="globe-source"><span className="status-dot"/>{status}<span className="source-divider">/</span>{renderer}</div>{pick&&<div className="pick-prompt">Select {pick==='stop'?'a science waypoint':`the ${pick} point`} on the terrain <button onClick={()=>useMission.getState().set({pick:null})}>Cancel</button></div>}</div>;
}
