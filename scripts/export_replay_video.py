"""Export a scientific visualization of actual NASA archive imagery.

Left: original camera pixels + a collision-checked local stereo route, if verified.
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
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend import data, replay
from backend import local_navigation

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



ANCHOR=(.5,.985);ANCHOR_HALF=.05


def route_strips(ribbons):
    """Join consecutive projected ribbon quads [l0,r0,r1,l1] into continuous strips (same rule as the web overlay)."""
    out=[];cur=None
    for q in ribbons:
        l0,r0,r1,l1=(tuple(v) for v in q[:4])
        joined=cur is not None and math.hypot(cur[0][-1][0]-l0[0],cur[0][-1][1]-l0[1])<.004
        if not joined:
            cur=([l0],[r0]);out.append(cur)
        cur[0].append(l1);cur[1].append(r1)
    return [s for s in out if len(s[0])>=2]


def _mid(a,b):return ((a[0]+b[0])/2,(a[1]+b[1])/2)


def _unit(v):
    n=math.hypot(*v) or 1.;return (v[0]/n,v[1]/n)


def rover_link(first):
    """Image-plane Bezier from the rover (bottom centre) to the first visible ground point, tangent to the route."""
    left,right=first;m0=_mid(left[0],right[0]);k=min(3,len(left)-1);m1=_mid(left[k],right[k])
    d=_unit((m1[0]-m0[0],m1[1]-m0[1]));dist=math.hypot(m0[0]-ANCHOR[0],m0[1]-ANCHOR[1])
    p1=(ANCHOR[0],ANCHOR[1]-dist*.45);p2=(m0[0]-d[0]*dist*.45,m0[1]-d[1]*dist*.45)
    w1=math.hypot(left[0][0]-right[0][0],left[0][1]-right[0][1])/2;ls=[];rs=[]
    for i in range(29):
        t=i/28;u=1-t
        x=u**3*ANCHOR[0]+3*u*u*t*p1[0]+3*u*t*t*p2[0]+t**3*m0[0]
        y=u**3*ANCHOR[1]+3*u*u*t*p1[1]+3*u*t*t*p2[1]+t**3*m0[1]
        dx=3*u*u*(p1[0]-ANCHOR[0])+6*u*t*(p2[0]-p1[0])+3*t*t*(m0[0]-p2[0])
        dy=3*u*u*(p1[1]-ANCHOR[1])+6*u*t*(p2[1]-p1[1])+3*t*t*(m0[1]-p2[1])
        n=_unit((-dy,dx));w=ANCHOR_HALF*(1-t)+w1*t
        ls.append((x+n[0]*w,y+n[1]*w));rs.append((x-n[0]*w,y-n[1]*w))
    return ls+rs[::-1]


def route_arrow(last):
    left,right=last
    if len(left)<3:return None
    n=len(left)-1;tip=_mid(left[n],right[n]);prev=_mid(left[n-2],right[n-2])
    d=_unit((tip[0]-prev[0],tip[1]-prev[1]));w=max(.012,math.hypot(left[n][0]-right[n][0],left[n][1]-right[n][1])*1.1);nr=(-d[1],d[0])
    return [(tip[0]+d[0]*w*1.6,tip[1]+d[1]*w*1.6),(tip[0]+nr[0]*w,tip[1]+nr[1]*w),(tip[0]-nr[0]*w,tip[1]-nr[1]*w)]


def draw_turn_cue(d,ox,oy,size,direction):
    """Amber in-place rotation arc at the rover position: a guidance icon, not a ground route."""
    dx=1 if direction=='right' else -1;cx,cy=ox+.5*size,oy+.9*size;rx,ry=.14*size,.1*size
    pts=[]
    for i in range(41):
        a=math.pi*i/40;pts.append((cx-dx*rx*math.cos(a),cy-ry*math.sin(a)))
    d.line(pts,fill=AMBER,width=9,joint='curve')
    ex=cx+dx*rx;d.polygon([(ex-dx*.028*size,cy-.025*size),(ex+dx*.012*size,cy),(ex-dx*.028*size,cy+.03*size)],fill=AMBER)


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
        d.text((30,58),'Real NASA footage / measured local stereo / archive map context',font=font(18),fill=MUTED)
        panel(d,(20,86,780,844),'REAL HAZCAM   /   LOCAL STEREO CLEARANCE CHECK')
        panel(d,(810,86,1580,844),'MARS MAP   /   HiRISE 25 cm ORTHOPHOTO')
        f=catalogue[st['frame_id']]
        camera=Image.open(data.DATA_DIR/(f.get('display_path') or f['image_path'])).convert('RGB').resize((size,size),Image.Resampling.LANCZOS)
        # Header is inside the panel but above the two square data viewports.
        cy=118;my=118
        canvas.paste(camera,(xcam,cy));canvas.paste(map_image,(xmap,my))
        d=ImageDraw.Draw(canvas)
        local=local_navigation.compute(st['frame_id'],result['goal'])
        for line in local.get('corridor_edges',[]):
            d.line([(xcam+p[0]*size,cy+p[1]*size) for p in line],fill=CYAN,width=2)
        # One union mask so the rover link, strips and arrow read as a single connected ribbon (no seams).
        px=lambda pts:[(p[0]*size,p[1]*size) for p in pts]
        parts=route_strips(local['ribbons']);mask=Image.new('L',(size,size),0);md=ImageDraw.Draw(mask)
        if parts:md.polygon(px(rover_link(parts[0])),fill=255)
        for left_edge,right_edge in parts:md.polygon(px(left_edge+right_edge[::-1]),fill=255)
        if parts and (arrow:=route_arrow(parts[-1])):md.polygon(px(arrow),fill=255)
        if parts:
            mask=mask.filter(ImageFilter.MaxFilter(3)).point(lambda v:int(v*.88))
            canvas.paste(Image.new('RGB',(size,size),CYAN),(xcam,cy),mask);d=ImageDraw.Draw(canvas)
        if local['status']=='turn_required' and local.get('turn_direction'):draw_turn_cue(d,xcam,cy,size,local['turn_direction'])
        for p in local.get('obstacles',[]):
            ox,oy=xcam+p[0]*size,cy+p[1]*size;d.rectangle((ox-2,oy-2,ox+2,oy+2),fill='#ff5a4a')
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
        if local['status']=='u_turn' and local['ribbons']:
            label=f"U-turn {local.get('turn_direction','')}: swing out, then head back"
        elif local['ribbons']:
            label=f"Local route / {local['min_obstacle_distance_m']:.1f} m from nearest rock" if local.get('min_obstacle_distance_m') else 'Local route / no rock in measured range'
        elif local['status']=='turn_required':
            turn=abs(local.get('required_turn_deg',0))
            label=f"Turn in place {local.get('turn_direction','')} {turn:.0f} deg"
        else:
            label='No measured safe corridor'
        d.text((xcam+27,cy+22),label,font=font(23,True),fill=CYAN if local['ribbons'] and local['status']!='u_turn' else AMBER)
        motion=st['motion'];details=motion['label']
        if motion['distance_m'] is not None:
            details+=f" / {motion['distance_m']:.1f} m / {motion['heading_change_deg']:+.1f} deg"
        d.text((xcam+27,cy+56),'Recorded: '+details,font=font(17),fill=WHITE)
        d.text((xcam+27,cy+81),f"Sol {st['sol']} / {st['time'][:19].replace('T',' ')} UTC",font=font(13),fill=MUTED)
        if not local['ribbons'] and local['status']!='turn_required':
            import textwrap
            box=(xcam+36,cy+230,xcam+size-36,cy+366)
            d.rounded_rectangle(box,radius=7,fill='#10252f')
            lines=textwrap.wrap(local['reason'],width=58)
            d.multiline_text((box[0]+16,box[1]+18),'\n'.join(lines),font=font(18),fill=WHITE,spacing=7)
        d.text((WIDTH-61,my+20),'N',font=font(20,True),fill=WHITE)
        d.text((30,866),'Accelerated archive timelapse. Recorded rover motion; calculated route does not control the rover.',font=font(16),fill=MUTED)
        d.text((30,886),'Local route: A* on measured stereo terrain for a 2.8 m wide rover, projected with the PDS camera model. Red: measured obstacles. Lines: rover width.',font=font(12),fill=MUTED)
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
           'camera_overlay':'Collision-checked local stereo A* route, metric ground ribbon projected with PDS CAHVORE; no decorative fallback',
           'local_navigation_settings':local_navigation.asdict(local_navigation.DEFAULT),
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
