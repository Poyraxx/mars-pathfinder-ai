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



ANCHOR=(.5,.985);PAINT=.55;HEAD_M=1.0;CHEVRON_EVERY_M=.9;CHEVRON_LEN_M=.28


def _unit(v):
    n=math.hypot(*v) or 1.;return (v[0]/n,v[1]/n)


def _smooth(a,k,lock=True):
    out=a.copy()
    for i in range(len(a)):
        if lock and (i==0 or i==len(a)-1):continue
        out[i]=a[max(0,i-k):i+k+1].mean(axis=0)
    return out


def ground_arrow(ribbons,ground_m=None,head=True):
    """Same geometry as src/routeGeometry.ts: one continuous perspective ribbon from the rover,
    width non-increasing with distance, gaps bridged; arrow head = last HEAD_M metres of ground and
    chevrons every CHEVRON_EVERY_M metres, so both shrink and foreshorten with distance."""
    if not ribbons:return None
    metric=ground_m is not None and len(ground_m)==len(ribbons)
    c=[];w=[];g=[]
    for i,q in enumerate(ribbons):
        l0,r0,r1,l1=(np.array(v,float) for v in q[:4]);a_=(l0+r0)/2;b_=(l1+r1)/2
        s0,s1=ground_m[i] if metric else (np.nan,np.nan)
        if not c or np.linalg.norm(c[-1]-a_)>1e-4:c.append(a_);w.append(np.linalg.norm(l0-r0)/2);g.append(s0)
        c.append(b_);w.append(np.linalg.norm(l1-r1)/2);g.append(s1)
    if len(c)<2:return None
    c=np.array(c);w=np.array(w);g=np.array(g,float);anchor=np.array(ANCHOR)
    k=min(4,len(c)-1);d0=_unit(c[k]-c[0]);d=np.linalg.norm(anchor-c[0]);startw=max(w[0],.03/PAINT)
    if d>.01:
        p1=anchor-[0,d*.45];p2=c[0]-np.array(d0)*d*.45;t=np.arange(24)[:,None]/24;u=1-t
        link=u**3*anchor+3*u*u*t*p1+3*u*t*t*p2+t**3*c[0]
        c=np.vstack([link,c]);w=np.r_[startw+(w[0]-startw)*t[:,0],w];g=np.r_[np.maximum(0,g[0]-.8*(1-t[:,0])),g]
    seg=np.linalg.norm(np.diff(c,axis=0),axis=1);s=np.r_[0,np.cumsum(seg)];keep=np.r_[True,seg>1e-9]
    c,w,g,s=c[keep],w[keep],g[keep],s[keep]
    grid=np.arange(0,s[-1],.006);grid=np.r_[grid,s[-1]] if s[-1]-grid[-1]>.0015 else grid
    c=np.column_stack([np.interp(grid,s,c[:,0]),np.interp(grid,s,c[:,1])]);w=np.interp(grid,s,w);g=np.interp(grid,s,g)
    c=_smooth(c,3);w=np.minimum.accumulate(w);w=np.maximum(_smooth(w[:,None],4,False)[:,0]*PAINT,.004)
    s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(c,axis=0),axis=1))];total=s[-1]
    if total<.03:return None
    idx=np.arange(len(c));tan=c[np.minimum(idx+2,len(c)-1)]-c[np.maximum(idx-2,0)]
    tan/=np.maximum(np.linalg.norm(tan,axis=1,keepdims=True),1e-9);nrm=np.column_stack([-tan[:,1],tan[:,0]])
    b=len(c)-1
    if not head:
        pass
    elif metric:
        while b>0 and g[-1]-g[b]<HEAD_M and total-s[b]<total*.4:b-=1
    else:
        head=min(total*.3,w[-1]*4.6)
        while b>0 and total-s[b]<head:b-=1
    left=c[:b+1]+nrm[:b+1]*w[:b+1,None];right=c[:b+1]-nrm[:b+1]*w[:b+1,None]
    ht=np.array(_unit(c[-1]-c[b]));nh=np.array([-ht[1],ht[0]]);hw=w[b]*2.0
    outline=[*map(tuple,left),tuple(c[b]+nh*hw),tuple(c[-1]),tuple(c[b]-nh*hw),*map(tuple,right[::-1])] if head else [*map(tuple,left),*map(tuple,right[::-1])]
    chevrons=[]
    def chevron(i,ln):
        ww=w[i]*.62;apex=c[i]+tan[i]*ln*.5;back=c[i]-tan[i]*ln*.5
        chevrons.append([tuple(back+nrm[i]*ww),tuple(apex),tuple(back-nrm[i]*ww)])
    if metric:
        nxt=math.ceil((g[0]+.4)/CHEVRON_EVERY_M)*CHEVRON_EVERY_M
        for i in range(1,b):
            if g[i]<nxt:continue
            j=min(i+3,len(c)-1);lo=max(i-3,0);dg=max(1e-4,g[j]-g[lo]);ln=min(w[i]*1.4,(s[j]-s[lo])/dg*CHEVRON_LEN_M)
            if g[-1]-g[i]>HEAD_M*.6:chevron(i,ln)
            nxt+=CHEVRON_EVERY_M
            while nxt<=g[i]:nxt+=CHEVRON_EVERY_M
    else:
        nxt=startw*PAINT*1.6
        for i in range(b):
            if s[i]<nxt:continue
            chevron(i,w[i]*.7);nxt=s[i]+w[i]*4.2
    return outline,chevrons,tuple(c[-1])



def far_route(segments,from_m):
    """Same rule as src/RouteOverlay.tsx: projected map route beyond the local corridor, smoothed in metres."""
    out=[]
    for seg in segments:
        seg=[p for p in seg if p[2]>=from_m]
        if len(seg)<4:continue
        sm=[]
        for i,p in enumerate(seg):
            if i in (0,len(seg)-1):sm.append(p);continue
            half=max(1.5,p[2]*.06);win=[q for q in seg if abs(q[2]-p[2])<=half]
            sm.append([sum(q[0] for q in win)/len(win),sum(q[1] for q in win)/len(win),p[2]])
        length=sum(math.hypot(sm[i][0]-sm[i-1][0],sm[i][1]-sm[i-1][1]) for i in range(1,len(sm)))
        if length>=.02:out.append(sm)
    return {'visible':out}

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
    locals_by_frame=local_navigation.compute_sequence(result['steps'],result['goal'])
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
        local=locals_by_frame[st['frame_id']]
        for line in local.get('corridor_edges',[]):
            d.line([(xcam+p[0]*size,cy+p[1]*size) for p in line],fill=CYAN,width=2)
        # Route painted on the ground: one continuous outline, depth-faded fill, light edge, surface chevrons.
        arrow=ground_arrow(local['ribbons'],local.get('ribbon_s_m'),head=False)
        if arrow:
            outline,chevrons,tip=arrow;sc=lambda pts:[(p[0]*size,p[1]*size) for p in pts]
            mask=Image.new('L',(size,size),0);ImageDraw.Draw(mask).polygon(sc(outline),fill=255)
            ytop=min(p[1] for p in outline)*size;ramp=np.clip((np.arange(size)-ytop)/max(1,size-ytop),0,1)
            alpha=(np.asarray(mask,np.float32)/255)*(.34+.32*ramp[:,None])
            region=np.asarray(canvas.crop((xcam,cy,xcam+size,cy+size)),np.float32)
            cyan=np.array([60,225,255],np.float32);blend=region+(255-region)*(cyan/255)*alpha[...,None]*1.6
            canvas.paste(Image.fromarray(np.clip(blend,0,255).astype(np.uint8)),(xcam,cy));d=ImageDraw.Draw(canvas)
            d.line([(xcam+x,cy+y) for x,y in sc(outline+[outline[0]])],fill='#c9f8ff',width=2,joint='curve')
            for ch in chevrons:d.line([(xcam+x,cy+y) for x,y in sc(ch)],fill='#f2fdff',width=3,joint='curve')
        # Far part of the plan (map route projected with the camera model), thin line ending in a dot.
        far=far_route((st.get('camera_route') or {}).get('segments') or [],(local.get('length_m') or 0)+1 if arrow else 2)
        P=lambda q:(xcam+q[0]*size,cy+q[1]*size)
        if far['visible']:
            if arrow and math.hypot(tip[0]-far['visible'][0][0][0],tip[1]-far['visible'][0][0][1])<.45:
                a,b=P(tip),P(far['visible'][0][0]);n=max(2,int(math.hypot(b[0]-a[0],b[1]-a[1])/9))
                for k in range(n+1):x,y=a[0]+(b[0]-a[0])*k/n,a[1]+(b[1]-a[1])*k/n;d.ellipse((x-1.6,y-1.6,x+1.6,y+1.6),fill=CYAN)
            for seg in far['visible']:
                pts=[P(q) for q in seg];d.line(pts,fill='#04121a',width=6,joint='curve');d.line(pts,fill=CYAN,width=3,joint='curve')
        end=P(far['visible'][-1][-1]) if far['visible'] else (P(tip) if arrow else None)
        if end:d.ellipse((end[0]-9,end[1]-9,end[0]+9,end[1]+9),fill='#1c6b7a');d.ellipse((end[0]-5,end[1]-5,end[0]+5,end[1]+5),fill='#f2fdff',outline=CYAN,width=2)
        for ring in local.get('distant_candidates',[]):d.line([P(q) for q in ring['ring']]+[P(ring['ring'][0])],fill='#ffb23d',width=2)
        for q in local.get('tracked_obstacles',[]):x,y=P(q);d.rectangle((x-2,y-2,x+2,y+2),fill='#ffb23d')
        if local['status']=='turn_required' and local.get('turn_direction'):draw_turn_cue(d,xcam,cy,size,local['turn_direction'])
        for p in local.get('obstacles',[]):
            ox,oy=xcam+p[0]*size,cy+p[1]*size;d.rectangle((ox-2,oy-2,ox+2,oy+2),fill='#ff5a4a')
        # Same route payload as the in-app map; only the cartographic projection changes.
        def mp(p):
            x,y=map_point(p);return x,y+(my-ymain)
        # Actual driven path (PLACES drive log), red; straight pose-to-pose only where no log exists.
        trail=[]
        for j,s_ in enumerate(result['steps'][:index+1]):
            pts=[{'lat':q[0],'lon':q[1]} for q in s_.get('driven') or []] or [result['trail'][j]]
            trail+=[mp(p) for p in pts]
        if len(trail)>1:d.line(trail,fill='#2a0805',width=7,joint='curve');d.line(trail,fill='#ff4a3d',width=4,joint='curve')
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
        if motion.get('detail'):
            details=motion['label']
        elif motion['distance_m'] is not None:
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
        d.text((30,886),'Near: A* on measured stereo for a 2.8 m rover. Thin line: map plan to destination. Red: measured obstacles; orange: remembered rocks / distant candidates.',font=font(12),fill=MUTED)
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
