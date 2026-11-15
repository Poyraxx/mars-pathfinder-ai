"""Model-free stereo proposals for visual adjudication, never automatic truth.

Only existing aligned XYZ/XYE measurements are used. Missing/far/uncertain
measurements stay unknown. These local-relief heuristics are weak supervision,
not certified rock heights or proof of safe driving.
"""
import concurrent.futures,hashlib,json,os,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training/live-demo-v12'
from backend.local_navigation import frame_basis
def grounded(m,sl,xyz,valid):
    h,w=valid.shape;r0,c0=sl[0].start,sl[1].start
    if r0<=2 or c0<=2 or sl[0].stop>=h-2 or sl[1].stop>=w-2:return False
    checked=touching=0
    for j in range(m.shape[1]):
        rows=np.flatnonzero(m[:,j])
        if not len(rows):continue
        r,c=r0+rows[-1],c0+j;below=r+2
        if below>=h or not valid[below,c] or not valid[r,c]:continue
        checked+=1;touching+=np.linalg.norm(xyz[below,c]-xyz[r,c])<=.35
    return checked>=3 and touching>=.5*checked
def prepare(args):
    src,frame=args;xyz=np.load(src['xyz_path']);xye=np.load(R/'data/prepared'/frame['xye_path']);err=np.linalg.norm(xye,axis=-1)
    valid=np.isfinite(xyz).all(-1)&(xyz!=0).any(-1)&np.isfinite(err)&(err<=.08)
    if frame.get('camera_model'):
        origin,forward,right=frame_basis(frame);delta=xyz[...,:2]-origin[:2];x,y=delta@forward,delta@right
        valid&=(np.hypot(x,y)<=6)&(x>.5)&~((x<1.6)&(np.abs(y)<=1.6))
    else:
        origin=np.asarray(frame['places']['telemetry_rover_origin_in_site_m']);x,y=xyz[...,0]-origin[0],xyz[...,1]-origin[1];valid&=(np.hypot(x,y)<=8)&(np.hypot(x,y)>1.6)
    shape=valid.shape;components=np.zeros(shape,np.uint16);flat=np.zeros(shape,bool);relief=np.full(shape,np.nan);objects=[]
    if valid.sum()>=100:
        gx=np.zeros(shape,int);gy=np.zeros(shape,int);gx[valid]=np.floor((x[valid]-x[valid].min())/.1).astype(int);gy[valid]=np.floor((y[valid]-y[valid].min())/.1).astype(int)
        low=np.full((gx.max()+1,gy.max()+1),np.inf);np.minimum.at(low,(gx[valid],gy[valid]),-xyz[...,2][valid]);seen=np.isfinite(low)
        indices=ndimage.distance_transform_edt(~seen,return_distances=False,return_indices=True);filled=low[tuple(indices)];ground=ndimage.median_filter(filled,size=15,mode='nearest');support=ndimage.uniform_filter(seen.astype(float),size=15)*225>=6
        relief[valid]=-xyz[...,2][valid]-ground[gx[valid],gy[valid]];valid&=support[gx,gy]
        raised=valid&(relief>=.10);cc,n=ndimage.label(raised)
        for cid,sl in enumerate(ndimage.find_objects(cc),1):
            if sl is None:continue
            m=cc[sl]==cid;px=int(m.sum());peak=float(np.nanmax(relief[sl][m]));points=xyz[sl][m];width=float(np.max(np.ptp(points[:,:2],axis=0)))
            if px<6 or peak>1.5 or width>2.0 or not grounded(m,sl,xyz,valid):continue
            label=len(objects)+1;components[sl][m]=label;yy,xx=np.where(m);cx=float(xx.mean()+sl[1].start);cy=float(yy.mean()+sl[0].start)
            objects.append({'id':label,'pixels':px,'peak_local_relief_m':peak,'extent_m':width,'centroid_native_geometry':[cx,cy],'bbox_geometry':[sl[1].start,sl[0].start,sl[1].stop,sl[0].stop]})
        flat=valid&(np.abs(relief)<=.035)&~ndimage.binary_dilation(raised,iterations=2)
        flat=ndimage.binary_erosion(flat,iterations=1)
    path=T/'geometry'/f"{src['index']:03}.npz";path.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(path,components=components,flat=flat,relief=relief.astype('float32'),valid=valid)
    return {'index':src['index'],'frame_id':src['frame_id'],'proposal_path':str(path),'proposal_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'geometry_shape':list(shape),'flat_proposal_pixels':int(flat.sum()),'usable_stereo_pixels':int(valid.sum()),'objects':objects,'manual_reviewed':False,'accepted_components':[]}
def main():
    sources=json.loads((T/'sources.json').read_text('utf8'))['entries'];frames={f['id']:f for f in json.loads((R/'data/prepared/frames.json').read_text('utf8'))};out=R/'artifacts/live-demo-v12-stereo';out.mkdir(exist_ok=True,parents=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(prepare,[(s,frames[s['frame_id']]) for s in sources]))
    font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',16);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12);tiles=[]
    for src,row in zip(sources,rows):
        raw=np.asarray(Image.open(src['image']).convert('RGB'));display=Image.fromarray((255*(raw/255.)**(.6 if np.median(raw)<45 else 1)).astype('uint8')).resize((512,512),Image.Resampling.BILINEAR);rgb=np.asarray(display).copy();g=np.load(row['proposal_path']);shape=g['components'].shape
        flat=np.asarray(Image.fromarray(g['flat']).resize((512,512),Image.Resampling.NEAREST));comp=np.asarray(Image.fromarray(g['components']).resize((512,512),Image.Resampling.NEAREST));rgb[flat]=(rgb[flat]*.7+[0,60,20]).astype('uint8');rgb[comp>0]=(rgb[comp>0]*.45+[140,20,10]).astype('uint8');display=Image.fromarray(rgb);draw=ImageDraw.Draw(display)
        for o in row['objects']:
            x,y=o['centroid_native_geometry'];x=x*512/shape[1];y=y*512/shape[0];draw.text((x,y),str(o['id']),font=font,fill='#ffff40',stroke_width=2,stroke_fill='black',anchor='mm')
        for v in range(64,512,64):draw.text((v+2,2),str(v),font=small,fill='#67ecff');draw.text((2,v),str(v),font=small,fill='#67ecff')
        tile=Image.new('RGB',(512,544),'#101820');tile.paste(display,(0,32));ImageDraw.Draw(tile).text((4,5),f"#{src['index']} Sol{src['sol']} / {len(row['objects'])} candidates",font=font,fill='white');tiles.append(tile)
    for start in range(0,len(tiles),4):
        sheet=Image.new('RGB',(1024,1088),'#101820')
        for n,tile in enumerate(tiles[start:start+4]):sheet.paste(tile,((n%2)*512,(n//2)*544))
        sheet.save(out/f'page-{start//4:02}.jpg',quality=96)
    record={'version':12,'source':'Existing aligned PDS XYZ/XYE, no model predictions','status':'proposals pending individual visual adjudication','policy':{'cell_m':.1,'ground_window_m':1.5,'max_error_m':.08,'hazcam_range_m':6,'navcam_range_m':8,'raised_relief_m':.10,'flat_abs_relief_m':.035,'requires_ground_contact_heuristic':True},'frames':rows}
    (T/'stereo-proposals.json').write_text(json.dumps(record,indent=2),encoding='utf8');print(json.dumps({'frames':len(rows),'proposed_components':sum(len(r['objects']) for r in rows),'frames_without_component':sum(not r['objects'] for r in rows),'flat_geometry_pixels':sum(r['flat_proposal_pixels'] for r in rows)}),flush=True)
if __name__=='__main__':main()
