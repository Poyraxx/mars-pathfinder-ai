import {useState} from 'react';
import {ArrowUpRight,RotateCcw} from 'lucide-react';
import {useMission} from './store';

/** Original NASA films: no invented per-frame pose or planner overlay. */
export const ARCHIVE_FILMS=[{
 id:'perseverance-sol200',title:'Perseverance · an autonomous drive',
 summary:'175 m across Jezero · 192 video frames · 29 seconds',
 url:'https://assets.science.nasa.gov/content/dam/science/psd/photojournal/pia/pia24/pia24934/PIA24934.mp4',
 source:'https://science.nasa.gov/photojournal/perseverances-sol-200-auto-nav-drive/',
 date:'12 September 2021 · Sol 200',
 note:'A real Navcam recording of a 175 m drive, including 167 m under AutoNav. NASA accelerated the images roughly 200×; source acquisitions are about 30 seconds apart.',
},{
 id:'curiosity-five-years',title:'Curiosity · five years on the ground',
 summary:'Front Hazcam + NASA location map · 5 minutes 49 seconds',
 url:'https://assets.science.nasa.gov/content/dam/science/psd/mars/downloadable_items/3/2/32074_RoverPOVFiveYearsOfCuriosityOnMars-1280.m4v',
 source:'https://science.nasa.gov/resource/rover-pov-five-years-of-curiosity-on-mars/',
 date:'2012–2017 · Gale Crater',
 note:'NASA’s full five-year Front Hazcam timelapse. The date, sol and rover location map are part of the original film. Film frames include stationary science work as well as drives.',
}];

export default function ArchiveFilm(){
 const s=useMission();
 const film=ARCHIVE_FILMS.find(f=>f.id===s.archiveVideo)??ARCHIVE_FILMS[0];
 const [failed,setFailed]=useState(false),[attempt,setAttempt]=useState(0);
 return <section className="archive-film" aria-label="NASA original drive film">
  <header className="film-heading"><div><h2>{film.title}</h2><p>{film.date} · NASA/JPL-Caltech</p></div><a className="ghost-button" href={film.source} target="_blank" rel="noreferrer">NASA source<ArrowUpRight size={14}/></a></header>
  <div className="film-screen">
   <video key={`${film.id}-${attempt}`} controls playsInline preload="metadata" src={film.url} onLoadedMetadata={()=>setFailed(false)} onError={()=>setFailed(true)} aria-label={film.title}/>
   {failed&&<div className="film-error" role="alert"><p>The NASA video could not load.</p><button className="ghost-button" onClick={()=>{setFailed(false);setAttempt(attempt+1);}}><RotateCcw size={14}/>Retry video</button><a href={film.source} target="_blank" rel="noreferrer">Open on NASA</a></div>}
  </div>
  <footer><p>{film.note}</p><p className="fine-print">Original NASA film. Interactive routes and calibrated camera overlays are available under Route replays.</p></footer>
 </section>;
}
