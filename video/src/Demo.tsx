import React from 'react';
import {AbsoluteFill, Audio, Img, OffthreadVideo, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig, Easing} from 'remotion';
import timing from '../public/vo/timing.json';

/** Team details shown in the closing card. Fill in before rendering; empty values are not shown. */
export const demoSchema = {team: '', members: '', url: 'mars-pathfinder-ai.onrender.com'};

const CYAN = '#3ce1ff', AMBER = '#f2a65a', INK = '#070b10', TEXT = '#eef4f6', MUTED = '#9fb2bd';
const FONT = '"Segoe UI Variable Display","Segoe UI",Inter,system-ui,sans-serif';
const sec = (s: number) => Math.round(s * 30);

// Scene boundaries (seconds). Voice lines start 0.35 s into their scene.
const S = [0, 3.3, 8.0, 14.0, 19.6, 25.6, 29.8];
const VO_OFFSET = 0.35;

const ease = (f: number, a: number, b: number, from = 0, to = 1) =>
  interpolate(f, [a, b], [from, to], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(.2, .7, .1, 1)});

const Sequenced: React.FC<{dir: string; count: number; duration: number}> = ({dir, count, duration}) => {
  const f = useCurrentFrame();
  const i = Math.min(count - 1, Math.floor(f / duration * count));
  return <Img src={staticFile(`capture/${dir}/${String(i).padStart(3, '0')}.jpg`)} style={{width: '100%', height: '100%', objectFit: 'cover'}} />;
};

const DataTag: React.FC<{children: React.ReactNode}> = ({children}) => {
  const f = useCurrentFrame();
  return <div style={{position: 'absolute', top: 44, left: 56, display: 'flex', gap: 10, alignItems: 'center', padding: '10px 18px', borderRadius: 999,
    background: 'rgba(7,11,16,.72)', border: '1px solid rgba(60,225,255,.35)', color: TEXT, fontFamily: FONT, fontSize: 22, letterSpacing: .3,
    opacity: ease(f, 4, 14), transform: `translateY(${ease(f, 4, 14, -12, 0)}px)`}}>
    <span style={{width: 9, height: 9, borderRadius: 9, background: CYAN, boxShadow: `0 0 12px ${CYAN}`}} />{children}
  </div>;
};

const Vignette = () => <AbsoluteFill style={{background: 'radial-gradient(ellipse at center, rgba(0,0,0,0) 55%, rgba(0,0,0,.55) 100%)'}} />;

const Subtitle: React.FC<{text: string; frames: number}> = ({text, frames}) => {
  const f = useCurrentFrame();
  const o = Math.min(ease(f, 0, 6), 1 - ease(f, frames - 6, frames));
  return <div style={{position: 'absolute', bottom: 70, left: 0, right: 0, display: 'flex', justifyContent: 'center', opacity: o}}>
    <div style={{maxWidth: 1400, padding: '12px 26px', borderRadius: 12, background: 'rgba(5,8,12,.78)', color: '#fff', fontFamily: FONT, fontSize: 38, fontWeight: 500, textAlign: 'center', lineHeight: 1.25}}>{text}</div>
  </div>;
};

const Title: React.FC = () => {
  const f = useCurrentFrame(); const {fps} = useVideoConfig();
  const s = spring({frame: f - 6, fps, config: {damping: 200}});
  return <AbsoluteFill style={{justifyContent: 'center', alignItems: 'flex-start', padding: '0 120px', fontFamily: FONT}}>
    <div style={{opacity: s, transform: `translateY(${(1 - s) * 30}px)`}}>
      <div style={{color: CYAN, fontSize: 26, letterSpacing: 6, fontWeight: 600}}>NASA SPACE APPS CHALLENGE 2026</div>
      <div style={{color: TEXT, fontSize: 124, fontWeight: 800, letterSpacing: -2, lineHeight: 1, marginTop: 14, textShadow: '0 6px 40px rgba(0,0,0,.7)'}}>Mars Pathfinder <span style={{color: CYAN}}>AI</span></div>
      <div style={{color: MUTED, fontSize: 32, marginTop: 18}}>Interplanetary Survival Guide: Martian Map</div>
    </div>
  </AbsoluteFill>;
};

const ScaleGap: React.FC<{frames: number}> = ({frames}) => {
  const f = useCurrentFrame();
  const cut = Math.round(frames * .5);
  return <AbsoluteFill>
    {f < cut ? <AbsoluteFill style={{transform: `scale(${1 + f / frames * .12})`}}><Sequenced dir="gale" count={50} duration={cut} /></AbsoluteFill>
      : <AbsoluteFill style={{background: INK}}>
        <OffthreadVideo src={staticFile('traverse.mp4')} startFrom={sec(3.6)} muted style={{position: 'absolute', width: 4208, height: 2367, left: -79 - (f - cut) * 1.2, top: -560 - (f - cut) * .5}} />
      </AbsoluteFill>}
    <Vignette />
    <div style={{position: 'absolute', right: 70, top: 300, fontFamily: FONT, textAlign: 'right'}}>
      {f < cut ? <><div style={{color: MUTED, fontSize: 30}}>Orbital terrain</div><div style={{color: TEXT, fontSize: 110, fontWeight: 800}}>200 m</div><div style={{color: MUTED, fontSize: 28}}>per pixel · MOLA/HRSC</div></>
        : <><div style={{color: MUTED, fontSize: 30}}>Ground hazard</div><div style={{color: AMBER, fontSize: 110, fontWeight: 800}}>25 cm</div><div style={{color: MUTED, fontSize: 28}}>real Curiosity Hazcam · stereo</div></>}
    </div>
    <DataTag>{f < cut ? 'Data: MOLA/HRSC global DEM, exaggerated ×12' : 'Data: MSL Front Hazcam (PDS) + stereo XYZ'}</DataTag>
  </AbsoluteFill>;
};

const Plan: React.FC<{frames: number}> = ({frames}) => {
  const f = useCurrentFrame();
  const a = Math.round(frames * .42), b = Math.round(frames * .72);
  const which = f < a ? 'route' : f < b ? 'dust' : 'geology';
  const local = f < a ? f / a : f < b ? (f - a) / (b - a) : (f - b) / (frames - b);
  const zoom = which === 'route' ? 1.6 + local * .12 : which === 'dust' ? 1.5 + local * .08 : 1.42 + local * .06;
  const origin = which === 'route' ? '54% 66%' : '70% 36%';
  const chips = [['Science stops', 'Sourced Curiosity targets'], ['Round trip', 'Return + time reserve'], ['Dust', 'MGS TES dust index'], ['Geology', 'USGS SIM 3292']];
  return <AbsoluteFill style={{background: INK}}>
    <AbsoluteFill style={{transform: `scale(${zoom})`, transformOrigin: origin}}>
      <Img src={staticFile(`capture/plan/${which}.jpg`)} style={{width: '100%', height: '100%', objectFit: 'cover'}} />
    </AbsoluteFill>
    <Vignette />
    <div style={{position: 'absolute', left: 56, top: 130, display: 'grid', gap: 14, fontFamily: FONT}}>
      {chips.map(([t, d], i) => {
        const o = ease(f, 8 + i * 16, 20 + i * 16);
        return <div key={t} style={{opacity: o, transform: `translateX(${(1 - o) * -30}px)`, padding: '14px 22px', borderRadius: 14, background: 'rgba(7,11,16,.82)', border: `1px solid ${i === 1 ? AMBER : CYAN}55`, minWidth: 380}}>
          <div style={{color: TEXT, fontSize: 32, fontWeight: 700}}>{t}</div><div style={{color: MUTED, fontSize: 22}}>{d}</div>
        </div>;
      })}
    </div>
    {which === 'route' && <div style={{position: 'absolute', right: 56, width: 980, top: 130, opacity: ease(f, 30, 42), transform: `translateY(${ease(f, 30, 42, 30, 0)}px)`}}>
      <Img src={staticFile('capture/plan/eva.png')} style={{width: '100%', borderRadius: 14, boxShadow: '0 20px 60px rgba(0,0,0,.6)'}} />
    </div>}
    <DataTag>{which === 'route' ? 'Data: HiRISE 1 m DEM · science stops at MSL PLACES positions' : which === 'dust' ? 'Data: MGS TES Dust Cover Index (Ruff & Christensen 2002)' : 'Data: USGS Geologic Map of Mars (SIM 3292)'}</DataTag>
  </AbsoluteFill>;
};

const Rover: React.FC<{frames: number}> = ({frames}) => {
  const f = useCurrentFrame();
  const z = 1 + f / frames * .06;
  return <AbsoluteFill style={{background: INK}}>
    <AbsoluteFill style={{transform: `scale(${z})`}}>
      <OffthreadVideo src={staticFile('traverse.mp4')} startFrom={sec(2.6)} muted style={{position: 'absolute', width: 2133, height: 1200, left: -106, top: -150}} />
    </AbsoluteFill>
    <Vignette />
    <div style={{position: 'absolute', left: 56, top: 112, display: 'flex', gap: 14, fontFamily: FONT}}>
      {[['AI', 'flags rocks', CYAN], ['Stereo', 'confirms size', CYAN], ['A*', 'routes around', AMBER]].map(([a, b, c], i) =>
        <div key={a} style={{opacity: ease(f, 6 + i * 14, 16 + i * 14), padding: '10px 18px', borderRadius: 12, background: 'rgba(7,11,16,.85)', border: `1px solid ${c}66`, color: TEXT, fontSize: 26}}><b style={{color: c}}>{a}</b> {b}</div>)}
    </div>
    <DataTag>Data: real Curiosity Front Hazcam frames · HiRISE 25 cm map · MSL PLACES poses</DataTag>
  </AbsoluteFill>;
};

const Results: React.FC<{frames: number}> = ({frames}) => {
  const f = useCurrentFrame();
  const n = Math.round(ease(f, 8, 70) * 31);
  const stats = [['86', 'hand-reviewed Hazcam images'], ['142', 'real Curiosity frames replayed'], ['100', 'automated tests pass']];
  return <AbsoluteFill style={{background: `radial-gradient(ellipse at 30% 40%, #0f2a33 0%, ${INK} 65%)`, fontFamily: FONT}}>
    <div style={{position: 'absolute', left: 120, top: 190}}>
      <div style={{color: MUTED, fontSize: 32}}>Rocks found in scenes the model never saw</div>
      <div style={{display: 'flex', alignItems: 'baseline', gap: 24, marginTop: 6}}>
        <span style={{color: TEXT, fontSize: 230, fontWeight: 800, letterSpacing: -6, lineHeight: 1}}>{n}</span>
        <span style={{color: MUTED, fontSize: 90, fontWeight: 700}}>/ 59</span>
      </div>
      <div style={{color: MUTED, fontSize: 26, marginTop: 8, opacity: ease(f, 40, 55)}}>3 independent hand-labelled test sets · previous model: <b style={{color: AMBER}}>0 of 56</b> · rover hardware almost never mistaken for rock</div>
    </div>
    <div style={{position: 'absolute', right: 110, top: 200, display: 'grid', gap: 22}}>
      {stats.map(([v, l], i) => <div key={l} style={{opacity: ease(f, 20 + i * 12, 32 + i * 12), transform: `translateX(${ease(f, 20 + i * 12, 32 + i * 12, 40, 0)}px)`, padding: '18px 26px', borderRadius: 16, background: 'rgba(255,255,255,.04)', border: '1px solid rgba(60,225,255,.25)', minWidth: 460}}>
        <div style={{color: CYAN, fontSize: 64, fontWeight: 800, lineHeight: 1}}>{v}</div><div style={{color: TEXT, fontSize: 26}}>{l}</div>
      </div>)}
    </div>
  </AbsoluteFill>;
};

const Close: React.FC<{team: string; members: string; url: string; frames: number}> = ({team, members, url, frames}) => {
  const f = useCurrentFrame();
  const data = ['MOLA/HRSC DEM', 'HiRISE DTM + ortho', 'MSL Hazcam (PDS)', 'MSL PLACES', 'MGS TES', 'Odyssey THEMIS', 'USGS SIM 3292', 'AI4Mars'];
  return <AbsoluteFill style={{background: INK, fontFamily: FONT}}>
    <AbsoluteFill style={{opacity: .35, transform: `scale(${1.05 + f / frames * .05})`}}><Img src={staticFile('capture/globe/059.jpg')} style={{width: '100%', height: '100%', objectFit: 'cover'}} /></AbsoluteFill>
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', textAlign: 'center'}}>
      <div style={{color: TEXT, fontSize: 96, fontWeight: 800, letterSpacing: -2, opacity: ease(f, 0, 12)}}>Mars Pathfinder <span style={{color: CYAN}}>AI</span></div>
      <div style={{display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 12, maxWidth: 1500, marginTop: 30}}>
        {data.map((d, i) => <span key={d} style={{opacity: ease(f, 6 + i * 3, 14 + i * 3), padding: '10px 18px', borderRadius: 999, border: '1px solid rgba(60,225,255,.4)', color: TEXT, fontSize: 24, background: 'rgba(7,11,16,.7)'}}>{d}</span>)}
      </div>
      <div style={{color: MUTED, fontSize: 28, marginTop: 34, opacity: ease(f, 30, 42)}}>{[team && `Team ${team}`, members].filter(Boolean).join(' · ')}</div>
      <div style={{color: CYAN, fontSize: 30, marginTop: 8, opacity: ease(f, 34, 46)}}>{url}</div>
      <div style={{color: MUTED, fontSize: 20, marginTop: 18, opacity: ease(f, 40, 52)}}>Research prototype · not a certified safety system · NASA data, no endorsement implied</div>
    </AbsoluteFill>
  </AbsoluteFill>;
};

export const Demo: React.FC<typeof demoSchema> = ({team, members, url}) => {
  const len = (i: number) => sec(S[i + 1]) - sec(S[i]);
  const scenes = [
    <AbsoluteFill key="0"><Sequenced dir="globe" count={60} duration={len(0) + 20} /><Vignette /><AbsoluteFill style={{background: 'linear-gradient(90deg, rgba(7,11,16,.88) 0%, rgba(7,11,16,.55) 45%, rgba(7,11,16,0) 75%)'}} /><Title /></AbsoluteFill>,
    <ScaleGap key="1" frames={len(1)} />,
    <Plan key="2" frames={len(2)} />,
    <Rover key="3" frames={len(3)} />,
    <Results key="4" frames={len(4)} />,
    <Close key="5" team={team} members={members} url={url} frames={len(5)} />,
  ];
  return <AbsoluteFill style={{background: INK}}>
    {scenes.map((node, i) => <Sequence key={i} from={sec(S[i])} durationInFrames={len(i)}>{node}</Sequence>)}
    {timing.lines.map((l, i) => <Sequence key={`vo${i}`} from={sec(S[i] + VO_OFFSET)} durationInFrames={sec(l.seconds) + 6}><Audio src={staticFile(l.file)} /></Sequence>)}
    {timing.lines.map((l, i) => <Sequence key={`sub${i}`} from={sec(S[i] + VO_OFFSET)} durationInFrames={Math.min(sec(l.seconds) + 12, sec(S[i + 1]) - sec(S[i] + VO_OFFSET))}><Subtitle text={l.en} frames={Math.min(sec(l.seconds) + 12, sec(S[i + 1]) - sec(S[i] + VO_OFFSET))} /></Sequence>)}
    <Audio src={staticFile('score.wav')} volume={f => interpolate(f, [0, 20, 850, 893], [0, .55, .55, 0], {extrapolateRight: 'clamp'})} />
  </AbsoluteFill>;
};
