"""Source-reviewed class-exclusion labels; never assign unmeasured ground free."""
import hashlib,json,os,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    m=json.loads((T/'live-demo-v12/manifest.json').read_text('utf8'));out=T/'live-demo-v13-constraints';out.mkdir(exist_ok=True,parents=True);entries=[]
    # Reviewed actual source contexts142..180 contain landscape and terrain
    # shadows, with no visible rover hardware. This excludes only class2.
    # It does NOT decide rock size, ground clearance or ground-vs-unknown.
    nav=list(range(142,181))
    clear={3:[[290,110,430,190],[140,110,205,160]],5:[[330,112,425,195],[170,102,225,153],[285,95,400,120]]}
    archive=R/'data/annotations/live-demo-v13-exclusion-masks.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for e in m['entries']:
            i=e['index'];w,h=e['shape'][::-1];paths={};hashes={};counts={}
            for c in [1,2]:
                mask=Image.new('L',(w,h));d=ImageDraw.Draw(mask)
                if c==2 and i in nav:d.rectangle((0,0,w-1,h-1),fill=1)
                if c==1:
                    for x0,y0,x1,y1 in clear.get(i,[]):d.rectangle((x0*w/512,y0*h/512,x1*w/512,y1*h/512),fill=1)
                p=out/f'{i:03}-not-class{c}.png';mask.save(p);paths[f'not_class{c}']=str(p);hashes[f'not_class{c}_sha256']=sha(p);counts[str(c)]=int(np.asarray(mask).sum());z.writestr(p.name,p.read_bytes())
            entries.append({'index':i,'frame_id':e['frame_id'],'image_sha256':e['image_sha256'],**paths,**hashes,'negative_pixel_counts':counts})
    policy={'version':13,'annotation_type':'Class exclusion only. A marked pixel is NOT class1 or NOT class2; other classes and physical clearance remain unknown.','all181_still_demo_fit_not_independent_test':True,'base_manifest_sha256':sha(T/'live-demo-v12/manifest.json'),'base_archive_sha256':sha(R/'data/annotations/live-demo-v12-masks.zip'),'all_navcam_hardware_absent_source_contexts_manually_reviewed':nav,'hazcam_clear_flat_appearance_not_rock_rectangles_512':clear,'review_evidence':'All native Navcam contexts reviewed in final composition sheets08..11, also source/crop sheets. Actual source3 and5 viewed with final labels and paired predictions, then flat soil interiors manually selected outside the boulder and equipment. No prediction pixels become truth.','no_new_metric_depth_or_driving_safety_label':True,'archive_sha256':sha(archive),'entries':entries}
    (out/'manifest.json').write_text(json.dumps(policy,indent=2),encoding='utf8');portable={**policy,'entries':[{k:v for k,v in e.items() if k not in ['not_class1','not_class2']} for e in entries]};(R/'data/annotations/live-demo-v13-exclusions.json').write_text(json.dumps(portable,indent=2),encoding='utf8');print('Sealed source-reviewed class exclusions for39 Navcam and2 Hazcam contexts',flush=True)
if __name__=='__main__':main()
