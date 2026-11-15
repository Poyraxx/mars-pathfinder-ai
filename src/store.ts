import {create} from 'zustand';
import type {ScienceTarget,Catalog,Point,Region,Mission,Event,Feature,OrbitalResult,Scenario,ReplayResult,LocalNavigation} from './types';
type State={
 catalog:Catalog|null; region:Region|null; start:Point|null;goal:Point|null;stops:Point[];pick:'start'|'goal'|'stop'|null;
 mission:Mission|null;events:Event[];error:string|null;busy:boolean;busyNote:string|null;followRover:boolean;view:'orbital'|'terrain'|'astronaut'|'dual';
 layer:string;showRisk:boolean;showTimeline:boolean;activeRoute:string;frameId:string;showSources:boolean;showAI:boolean;showNav:boolean;exploreTarget:(Point&{alt:number})|null;
 planMode:'local'|'global';features:Feature[];feature:Feature|null;orbital:OrbitalResult|null;avoid:string[];
 mode:'explore'|'plan'|'replay';scenarios:Scenario[];replay:ReplayResult|null;replayStep:number;replayView:'3d'|'navcam';cameraOverlay:'guidance'|'projection';replayGoalPick:boolean;replayImageReady:string|null;localNavigation:LocalNavigation|null;localBatch:{key:string;frames:Record<string,LocalNavigation>}|null;relief:boolean;
 overlay:string|null;profileKind:'astronaut'|'crew_cart'|'rover';stopWork:number[];budgetMin:number|null;evaDate:string;scienceTargets:{gale:ScienceTarget[];global:ScienceTarget[]}|null;stopNames:(string|null)[];
 set:(s:Partial<State>)=>void;selectRegion:(r:Region)=>void;selectPoint:(p:Point)=>void;
};
export const useMission=create<State>((set,get)=>({
 catalog:null,region:null,start:null,goal:null,stops:[],pick:null,mission:null,events:[],error:null,busy:false,busyNote:null,followRover:false,
 view:'orbital',layer:'viking',showRisk:false,showTimeline:window.innerHeight>=780,activeRoute:'surface',frameId:'',showSources:false,showAI:window.innerWidth>=1200,showNav:false,exploreTarget:null,planMode:'global',features:[],feature:null,orbital:null,avoid:[],mode:'plan',scenarios:[],replay:null,replayStep:0,replayView:'navcam',cameraOverlay:'guidance',replayGoalPick:false,replayImageReady:null,localNavigation:null,localBatch:null,relief:false,overlay:null,profileKind:'astronaut',stopWork:[],budgetMin:null,evaDate:new Date().toISOString().slice(0,10),scienceTargets:null,stopNames:[],
 set:(s)=>set(s),selectRegion:(r)=>set({planMode:'local',feature:null,region:r,start:r.default_start,goal:r.default_goal,stops:[],mission:null,events:[],error:null,view:'terrain',exploreTarget:null,showNav:false}),
 selectPoint:(p)=>{const {pick,stops}=get();if(get().replayGoalPick){set({replayGoalPick:false,pick:null});window.dispatchEvent(new CustomEvent('replay-goal',{detail:p}));return;}if(pick==='start')set({start:p,pick:null,mission:null,events:[]});else if(pick==='goal')set({goal:p,pick:null,mission:null,events:[]});else if(pick==='stop')set({stops:[...stops,p].slice(0,5),stopWork:[...get().stopWork.slice(0,stops.length),30].slice(0,5),stopNames:[...get().stopNames.slice(0,stops.length),null].slice(0,5),pick:null,mission:null,events:[]});}
}));
export async function api<T>(url:string,body?:unknown):Promise<T>{
 const response=await fetch(url,body!==undefined?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:undefined);
 if(!response.ok){const error=await response.json().catch(()=>({detail:{message:'Service temporarily unavailable'}}));throw Object.assign(new Error(error.detail?.message||error.detail||`Request failed (${response.status})`),{status:response.status});}
 return response.json();
}
if(import.meta.env.DEV)(window as unknown as {__mission:typeof useMission}).__mission=useMission;
