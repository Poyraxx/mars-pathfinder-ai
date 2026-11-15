"""Diagnose real-image test expectations with actual metric paths/projections."""
import argparse,json,os,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np
from PIL import Image,ImageDraw
from backend import data,local_navigation as local

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();os.environ['MARS_NAVIGATION_MODEL_PATH']=str(a.model);a.output.mkdir(exist_ok=True,parents=True);rows=[]
    for prefix in ['FLA_404772660','FLA_402284021']:
        frame=next(f for f in data.frame_catalog() if f['id'].startswith(prefix));result=local.compute(frame['id']);_,mask,conf,_,_=local.infer(frame['id']);xyz=np.load(data.DATA_DIR/frame['xyz_path']);xye=np.load(data.DATA_DIR/frame['xye_path']);origin,forward,right=local.frame_basis(frame);terrain=local.build_terrain(xyz,xye,mask,conf,origin,forward,right);path=np.asarray(result['path_xy']);clear=all(local.segment_clear(p,q,terrain) for p,q in zip(path,path[1:]));projection=local.project_path(frame,terrain,path) if len(path)>1 else {'segments':[],'ribbons':[]};heights=local._height(path,terrain) if len(path) else [];site=np.column_stack((origin[:2]+path[:,0,None]*forward+path[:,1,None]*right,-heights)) if len(path) else np.empty((0,3));uv=local.site_points_to_pixels(site,frame['camera_model']) if len(path) else []
        rows.append({'frame_id':frame['id'],'status':result['status'],'reason':result.get('reason'),'path_xy_m':path.tolist(),'all_path_segments_clear':clear,'minimum_obstacle_distance_m':result.get('min_obstacle_distance_m'),'geometry':result['geometry'],'projection_segments':len(projection['segments']),'projection_ribbons':len(projection['ribbons']),'path_pixel_coordinates':np.asarray(uv).tolist(),'source_shape':frame['camera_model']['native_shape']})
        source=Image.open(data.DATA_DIR/(frame.get('display_path') or frame['image_path'])).convert('RGB').resize((512,512));rgb=np.asarray(source).copy();classes=np.asarray(Image.fromarray(mask).resize((512,512),Image.Resampling.NEAREST));colors={3:[255,70,60],255:[50,150,255]}
        for c,color in colors.items():rgb[classes==c]=(.6*rgb[classes==c]+.4*np.asarray(color)).astype('uint8')
        projected=source.convert('RGBA');layer=Image.new('RGBA',source.size);d=ImageDraw.Draw(layer)
        for poly in result['ribbons']:d.polygon([(x*512,y*512) for x,y in poly],fill=(40,225,245,100))
        for x,y in result['obstacles']:d.ellipse((x*512-1,y*512-1,x*512+1,y*512+1),fill=(255,50,50,255))
        canvas=Image.new('RGB',(1536,550),'#101820');canvas.paste(source,(0,28));canvas.paste(Image.fromarray(rgb),(512,28));canvas.paste(Image.alpha_composite(projected,layer).convert('RGB'),(1024,28));ImageDraw.Draw(canvas).text((5,5),prefix+' source / runtime appearance (blue=self/unknown) / measured projection '+result['status'],fill='white');canvas.save(a.output/(prefix+'.jpg'),quality=95)
    (a.output/'report.json').write_text(json.dumps(rows,indent=2),encoding='utf8');print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
