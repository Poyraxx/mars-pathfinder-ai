import React from 'react';
import {AbsoluteFill, Audio, Img, OffthreadVideo, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig, Easing} from 'remotion';
import timing from '../public/vo/timing.json';

/** Team details shown at the start and end. */
export const demoSchema = {team: 'Asteria Roket', members: 'Erol Poyraz Çakmaz · Fatih Han Bayram · Ayşe Özsarı', url: 'mars-pathfinder-ai.onrender.com'};

const CYAN = '#3ce1ff', INK = '#070b10', TEXT = '#eef4f6', MUTED = '#9fb2bd';
const FONT = '"Segoe UI Variable Display","Segoe UI",Inter,system-ui,sans-serif';
const sec = (s: number) => Math.round(s * 30);

// Each scene = its voice line + an equal share of the remaining time (voice is never sped up).
export const TOTAL = 29.8;
const PAD = (TOTAL - timing.lines.reduce((a, l) => a + l.seconds, 0)) / timing.lines.length;
const S = timing.lines.reduce<number[]>((acc, l) => [...acc, acc[acc.length - 1] + l.seconds + PAD], [0]).map((v, i, a) => i === a.length - 1 ? TOTAL : Math.round(v * 30) / 30);
const VO_OFFSET = Math.min(0.4, PAD * .4);

const ease = (f: number, a: number, b: number, from = 0, to = 1) =>
  interpolate(f, [a, b], [from, to], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(.2, .7, .1, 1)});

const Frames: React.FC<{dir: string; count: number; duration: number}> = ({dir, count, duration}) => {
  const f = useCurrentFrame();
  const i = Math.min(count - 1, Math.floor(f / duration * count));
  return <Img src={staticFile(`capture/${dir}/${String(i).padStart(3, '0')}.jpg`)} style={{width: '100%', height: '100%', objectFit: 'cover'}} />;
};

/** A real screen recording of the app, played from `from` seconds at `rate`, with a slow push-in. */
const Rec: React.FC<{name: string; from: number; rate: number; frames: number; zoom: [number, number]; origin: string}> = ({name, from, rate, frames, zoom, origin}) => {
  const f = useCurrentFrame();
  return <AbsoluteFill style={{transform: `scale(${interpolate(f, [0, frames], zoom)})`, transformOrigin: origin}}>
    <OffthreadVideo src={staticFile(`rec/${name}.webm`)} startFrom={sec(from)} playbackRate={rate} muted style={{width: '100%', height: '100%'}} />
  </AbsoluteFill>;
};

/** Small source caption, one per scene. */
const Caption: React.FC<{children: React.ReactNode; at?: number}> = ({children, at = 4}) => {
  const f = useCurrentFrame();
  return <div style={{position: 'absolute', top: 40, right: 48, padding: '9px 16px', borderRadius: 8, background: 'rgba(7,11,16,.8)', color: MUTED,
    fontFamily: FONT, fontSize: 21, opacity: ease(f, at, at + 10)}}>{children}</div>;
};

const Shade = ({side = 'left'}: {side?: 'left' | 'bottom'}) => <AbsoluteFill style={{background: side === 'left'
  ? 'linear-gradient(90deg, rgba(7,11,16,.9) 0%, rgba(7,11,16,.55) 42%, rgba(7,11,16,0) 72%)'
  : 'linear-gradient(0deg, rgba(7,11,16,.75) 0%, rgba(7,11,16,0) 30%)'}} />;

const Subtitle: React.FC<{text: string; frames: number}> = ({text, frames}) => {
  const f = useCurrentFrame();
  const o = Math.min(ease(f, 0, 5), 1 - ease(f, frames - 5, frames));
  return <div style={{position: 'absolute', bottom: 60, left: 0, right: 0, display: 'flex', justifyContent: 'center', opacity: o}}>
    <div style={{maxWidth: 1500, padding: '10px 24px', borderRadius: 10, background: 'rgba(5,8,12,.8)', color: '#fff', fontFamily: FONT, fontSize: 36, fontWeight: 500, textAlign: 'center', lineHeight: 1.25}}>{text}</div>
  </div>;
};

const Rise: React.FC<{delay?: number; children: React.ReactNode}> = ({delay = 0, children}) => {
  const f = useCurrentFrame(); const {fps} = useVideoConfig();
  const s = spring({frame: f - delay, fps, config: {damping: 200}});
  return <div style={{opacity: s, transform: `translateY(${(1 - s) * 18}px)`}}>{children}</div>;
};

const Team: React.FC<{team: string; frames: number}> = ({team, frames}) => <AbsoluteFill>
  <Frames dir="globe" count={60} duration={frames + 40} /><Shade />
  <AbsoluteFill style={{justifyContent: 'center', padding: '0 120px', fontFamily: FONT}}>
    <Rise delay={4}><div style={{color: CYAN, fontSize: 24, letterSpacing: 5, fontWeight: 600}}>NASA SPACE APPS CHALLENGE 2026 · ANKARA</div></Rise>
    <Rise delay={9}><div style={{color: TEXT, fontSize: 96, fontWeight: 800, letterSpacing: -1.5, marginTop: 10}}>{team}</div></Rise>
    <Rise delay={16}><div style={{color: MUTED, fontSize: 32, marginTop: 8}}>Challenge: Interplanetary Survival Guide — Martian Map</div></Rise>
  </AbsoluteFill>
</AbsoluteFill>;

const Title: React.FC<{frames: number}> = ({frames}) => {
  const f = useCurrentFrame();
  return <AbsoluteFill>
    <AbsoluteFill style={{transform: `scale(${1 + f / frames * .08})`}}><Frames dir="gale" count={50} duration={frames} /></AbsoluteFill><Shade />
    <AbsoluteFill style={{justifyContent: 'center', padding: '0 120px', fontFamily: FONT}}>
      <Rise delay={3}><div style={{color: TEXT, fontSize: 120, fontWeight: 800, letterSpacing: -2, lineHeight: 1}}>Mars Pathfinder</div></Rise>
      <Rise delay={12}><div style={{color: MUTED, fontSize: 36, marginTop: 18, maxWidth: 900}}>A route map for the first astronaut to walk on Mars</div></Rise>
    </AbsoluteFill>
    <Caption at={20}>Gale crater · MOLA/HRSC elevation, relief ×12</Caption>
  </AbsoluteFill>;
};

// Line 3: relief → dust → geology (Conditions), then a planned route with science stops and the round-trip bar.
const Plan: React.FC<{frames: number}> = ({frames}) => {
  const cut = Math.round(frames * .42);
  return <AbsoluteFill style={{background: INK}}>
    <Sequence durationInFrames={cut}>
      <Rec name="conditions" from={3} rate={7.4 / (cut / 30)} frames={cut} zoom={[1.6, 1.7]} origin="64% 100%" />
      <Caption>Relief, dust (MGS TES), geology (USGS)</Caption>
    </Sequence>
    <Sequence from={cut}>
      <Rec name="plan" from={5.2} rate={9.6 / ((frames - cut) / 30)} frames={frames - cut} zoom={[1.08, 1.16]} origin="55% 70%" />
      <Caption>Science stops · round trip with reserve · HiRISE 1 m DEM</Caption>
    </Sequence>
    <Shade side="bottom" />
  </AbsoluteFill>;
};

const Rover: React.FC<{frames: number}> = ({frames}) => <AbsoluteFill style={{background: INK}}>
  <Rec name="replay" from={0.8} rate={10.6 / (frames / 30)} frames={frames} zoom={[1.1, 1.16]} origin="50% 42%" />
  <Shade side="bottom" />
  <Caption>Real Curiosity Front Hazcam frames · stereo-checked rocks</Caption>
</AbsoluteFill>;

const Close: React.FC<{team: string; members: string; url: string; frames: number}> = ({team, members, url, frames}) => {
  const f = useCurrentFrame();
  return <AbsoluteFill style={{background: INK, fontFamily: FONT}}>
    <AbsoluteFill style={{opacity: .3, transform: `scale(${1.04 + f / frames * .04})`}}><Img src={staticFile('capture/globe/059.jpg')} style={{width: '100%', height: '100%', objectFit: 'cover'}} /></AbsoluteFill>
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', textAlign: 'center'}}>
      <Rise delay={2}><div style={{color: TEXT, fontSize: 92, fontWeight: 800, letterSpacing: -1.5}}>Mars Pathfinder</div></Rise>
      <Rise delay={10}><div style={{color: MUTED, fontSize: 30, marginTop: 20}}><span style={{color: TEXT, fontWeight: 700}}>{team}</span> · Ankara<br />{members}</div></Rise>
      <Rise delay={16}><div style={{color: CYAN, fontSize: 30, marginTop: 14}}>{url}</div></Rise>
      <Rise delay={22}><div style={{color: MUTED, fontSize: 19, marginTop: 26}}>AI voiceover (ElevenLabs) · research prototype, not a safety system · NASA open data, no endorsement implied</div></Rise>
    </AbsoluteFill>
  </AbsoluteFill>;
};

export const Demo: React.FC<typeof demoSchema> = ({team, members, url}) => {
  const len = (i: number) => sec(S[i + 1]) - sec(S[i]);
  const scenes = [
    <Team key="0" team={team} frames={len(0)} />,
    <Title key="1" frames={len(1)} />,
    <Plan key="2" frames={len(2)} />,
    <Rover key="3" frames={len(3)} />,
    <Close key="4" team={team} members={members} url={url} frames={len(4)} />,
  ];
  const subLen = (i: number) => Math.min(sec(timing.lines[i].seconds) + 10, sec(S[i + 1]) - sec(S[i] + VO_OFFSET));
  return <AbsoluteFill style={{background: INK}}>
    {scenes.map((node, i) => <Sequence key={i} from={sec(S[i])} durationInFrames={len(i)}>{node}</Sequence>)}
    {timing.lines.map((l, i) => <Sequence key={`vo${i}`} from={sec(S[i] + VO_OFFSET)} durationInFrames={sec(l.seconds) + 6}><Audio src={staticFile(l.file)} /></Sequence>)}
    {timing.lines.map((l, i) => <Sequence key={`sub${i}`} from={sec(S[i] + VO_OFFSET)} durationInFrames={subLen(i)}><Subtitle text={l.en} frames={subLen(i)} /></Sequence>)}
    <Audio src={staticFile('score.wav')} volume={f => interpolate(f, [0, 30, sec(TOTAL) - 30, sec(TOTAL)], [0, .22, .22, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})} />
  </AbsoluteFill>;
};
