"""Export a scientific visualization of actual NASA archive imagery.

Left: original camera pixels + a rover-centred directional navigation HUD.
Right: actual georeferenced HiRISE ortho + the identical route and PLACES pose.
No synthetic camera frames, simulated rover movement or guessed camera FOV.
The sparse acquisitions are held as a clearly labelled accelerated timelapse.
Run: python scripts/export_replay_video.py [scenario_id] [--duration seconds]
Requires imageio-ffmpeg only at authoring time; exported films need no encoder on Render.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend import data, replay

WIDTH,HEIGHT=1600,900
FPS,DURATION=24,30
CYAN='#3ce1ff'
WHITE='#e2ebee'
MUTED='#a8bac5'
AMBER='#f2a65a'


def font(size,bold=False):
    win=Path('C:/Windows/Fonts')/('arialbd.ttf' if bold else 'arial.ttf')
    linux=Path('/usr/share/fonts/truetype/dejavu')/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')
    return ImageFont.truetype(str(win if win.exists() else linux),size)


def render_keyframes(result):
    meta=json.loads((data.DATA_DIR/'pds/region-ortho.json').read_text('utf-8'))
    ortho=Image.open(data.DATA_DIR/'pds'/meta['image']).convert('RGB')
    r=replay.R; catalogue={f['id']:f for f in data.frame_catalog()}
    points=[result['goal'],*result['trail'],*(p for s in result['steps'] for p in s['route'])]
    east=[math.radians(p['lon'])*r for p in points];north=[math.radians(p['lat'])*r for p in points]
    span=max(max(east)-min(east),max(north)-min(north),18)+16
    centre_e=(max(east)+min(east))/2;centre_n=(max(north)+min(north))/2
    left,top=centre_e-span/2,centre_n+span/2
    res=meta['pixel_resolution_m']
    box=((left-meta['upper_left_easting_m'])/res,(meta['upper_left_northing_m']-top)/res,
         (left+span-meta['upper_left_easting_m'])/res,(meta['upper_left_northing_m']-top+span)/res)
    if box[0]<0 or box[1]<0 or box[2]>ortho.width or box[3]>ortho.height:
        raise ValueError('Video map crop exceeds actual orthophoto coverage')
    size=730;map_image=ortho.crop(tuple(map(int,box))).resize((size,size),Image.Resampling.LANCZOS)
    xcam,ymain,xmap=30,102,840
    def map_point(p):
        return (xmap+(math.radians(p['lon'])*r-left)/span*size,ymain+(top-math.radians(p['lat'])*r)/span*size)
    def panel(draw,rect,label):
        draw.rounded_rectangle(rect,radius=8,fill='#0b151d',outline='#304350',width=1)
        draw.text((rect[0]+14,rect[1]+11),label,font=font(17,True),fill=WHITE)
    rendered=[]
    for index,st in enumerate(result['steps']):
        canvas=Image.new('RGB',(WIDTH,HEIGHT),'#070e14');d=ImageDraw.Draw(canvas)
        d.text((30,21),'PATHFINDER  /  ROVER NAVIGATION',font=font(27,True),fill=WHITE)
        d.text((30,58),'Real NASA footage + one shared route + recorded Mars position',font=font(18),fill=MUTED)
        panel(d,(20,86,780,844),'REAL HAZCAM   /   ROVER-CENTRED DIRECTION GUIDE')
        panel(d,(810,86,1580,844),'MARS MAP   /   HiRISE 25 cm ORTHOPHOTO')
        f=catalogue[st['frame_id']]
        camera=Image.open(data.DATA_DIR/(f.get('display_path') or f['image_path'])).convert('RGB').resize((size,size),Image.Resampling.LANCZOS)
        # Header is inside the panel but above the two square data viewports.
        cy=118;my=118
        canvas.paste(camera,(xcam,cy));canvas.paste(map_image,(xmap,my))
        d=ImageDraw.Draw(canvas)
        guide=st['guidance'];colour=AMBER if guide['kind']=='u_turn' else CYAN
        xy=[(xcam+p[0]*size,cy+p[1]*size) for p in guide['points']]
        d.line(xy,fill='#07131a',width=11,joint='curve');d.line(xy,fill=colour,width=6,joint='curve')
        tip=np.array(xy[-1]);direction=tip-np.array(xy[-3]);direction/=max(np.linalg.norm(direction),1e-6)
        side=np.array([-direction[1],direction[0]])
        d.polygon([tuple(tip+direction*8),tuple(tip-direction*16+side*10),tuple(tip-direction*16-side*10)],fill=colour)
        ax,ay=xcam+.5*size,cy+.94*size
        d.ellipse((ax-10,ay-10,ax+10,ay+10),fill=colour,outline='#07131a',width=3)
        d.polygon([(ax,ay-18),(ax-12,ay+8),(ax,ay+2),(ax+12,ay+8)],fill=WHITE,outline='#07131a',width=2)
        # Same route payload as the in-app map; only the cartographic projection changes.
        def mp(p):
            x,y=map_point(p);return x,y+(my-ymain)
        trail=[mp(p) for p in result['trail'][:index+1]]
        if len(trail)>1:d.line(trail,fill=WHITE,width=3)
        route=[mp(st['rover']),*[mp(p) for p in st['route']]]
        if len(route)>1:
            d.line(route,fill='#07131a',width=10,joint='curve');d.line(route,fill=CYAN,width=5,joint='curve')
        gx,gy=mp(result['goal']);d.ellipse((gx-7,gy-7,gx+7,gy+7),fill=AMBER,outline='#07131a',width=2)
        x,y=mp(st['rover']);yaw=math.radians(st['rover']['yaw_deg'])
        forward=np.array([math.sin(yaw),-math.cos(yaw)]);side=np.array([math.cos(yaw),math.sin(yaw)])
        centre=np.array([x,y]);triangle=[tuple(centre+forward*14),tuple(centre-forward*9+side*9),tuple(centre-forward*9-side*9)]
        d.polygon(triangle,fill=WHITE,outline='#10252f',width=2)
        d.rounded_rectangle((xmap+12,my+12,xmap+240,my+88),radius=7,fill='#0b151d')
        d.text((xmap+24,my+21),f"{st['remaining_m']:.1f} m remaining",font=font(24,True),fill=WHITE)
        d.text((xmap+24,my+55),f"ROUTE v{st['route_version']}  /  FRAME {index+1} OF {len(result['steps'])}",font=font(14),fill=MUTED)
        d.rounded_rectangle((xcam+12,cy+12,xcam+size-12,cy+100),radius=7,fill='#10252f')
        d.text((xcam+27,cy+22),guide['instruction'],font=font(23,True),fill=colour)
        motion=st['motion'];details=motion['label']
        if motion['distance_m'] is not None:
            details+=f" / {motion['distance_m']:.1f} m / {motion['heading_change_deg']:+.1f} deg"
        d.text((xcam+27,cy+56),'Recorded: '+details,font=font(17),fill=WHITE)
        d.text((xcam+27,cy+81),f"Sol {st['sol']} / {st['time'][:19].replace('T',' ')} UTC",font=font(13),fill=MUTED)
        d.text((WIDTH-61,my+20),'N',font=font(20,True),fill=WHITE)
        d.text((30,866),'Accelerated archive timelapse. Recorded rover motion; calculated route does not control the rover.',font=font(16),fill=MUTED)
        d.text((30,886),'Centre guide is a directional HUD, not a calibrated ground trace. Recorded motion is from PLACES pose differences.',font=font(12),fill=MUTED)
        rendered.append(canvas)
    return rendered


def main():
    parser=argparse.ArgumentParser();parser.add_argument('scenario',nargs='?',default='hazcam-rocknest-approach');parser.add_argument('--duration',type=int,default=DURATION);args=parser.parse_args()
    duration=max(1,args.duration)
    data.load_regions();result=replay.compute(args.scenario)
    frames=render_keyframes(result)
    output=data.DATA_DIR/'demos'/f'{args.scenario}.mp4';output.parent.mkdir(exist_ok=True)
    exe=imageio_ffmpeg.get_ffmpeg_exe()
    command=[exe,'-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-s',f'{WIDTH}x{HEIGHT}','-pix_fmt','rgb24','-r',str(FPS),'-i','-',
             '-an','-vcodec','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        payloads=[f.tobytes() for f in frames]
        for i in range(FPS*duration):process.stdin.write(payloads[min(len(frames)-1,i*len(frames)//(FPS*duration))])
    finally:
        process.stdin.close()
    if process.wait()!=0:raise RuntimeError('Video encoder failed')
    proof={'scenario':args.scenario,'duration_s':duration,'fps':FPS,'dimensions':[WIDTH,HEIGHT],
           'footage':'Actual NASA/JPL-Caltech PDS Hazcam pixels, accelerated still-image timelapse; no interpolated motion',
           'map':'Actual PLACES HiRISE 25 cm orthophoto','calculation':result['method'],
           'camera_overlay':'Rover-centred directional HUD from remaining route and heading; not a calibrated ground projection',
           'motion_analysis':'Independent changes between successive real PLACES poses, not planned motion',
           'rover_frames':[s['frame_id'] for s in result['steps']], 'events':result['events'],
           'input_frames_sha256':hashlib.sha256((data.DATA_DIR/'frames.json').read_bytes()).hexdigest(),
           'video_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
           'limitation':'Camera reprojection is validated; absolute orbital placement of these Hazcam poses remains archive-based, not independently accepted.'}
    output.with_suffix('.json').write_text(json.dumps(proof,indent=2),'utf-8')
    artifacts=ROOT/'artifacts';artifacts.mkdir(exist_ok=True)
    frames[1 if len(frames)>1 else 0].save(artifacts/f'{args.scenario}-preview.png')
    print(output,output.stat().st_size,flush=True)


if __name__=='__main__':main()
