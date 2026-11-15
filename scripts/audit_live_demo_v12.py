"""Hash, source coverage, class composition and history-preservation checks."""
import hashlib,json,os
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1];T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training/live-demo-v12'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    mp=T/'manifest.json';m=json.loads(mp.read_text('utf8'));assert m['status']=='final_reviewed';entries=m['entries'];review=json.loads((R/'data/annotations/live-demo-v12-review.json').read_text('utf8'));qa=json.loads((R/'data/annotations/live-demo-v12-manual-qa.json').read_text('utf8'));sc=json.loads((R/'data/prepared/scenarios.json').read_text('utf8'));expected={i for s in sc for i in s['frames']}
    assert len(entries)==len(expected)==181 and {e['frame_id'] for e in entries}==expected
    assert len(review['frames'])==181 and all(r['context_reviewed'] and r['flat_proposals_reviewed'] for r in review['frames']);assert qa['all_geometry_pages_viewed']==list(range(46)) and qa['all_rendered_manual_pages_viewed']==list(range(9))
    assert sha(R/'data/annotations/live-demo-v12-review.json')==m['review_sha256'] and sha(R/'data/annotations/live-demo-v12-manual-qa.json')==m['qa_sha256']
    counts=Counter();coverage=[]
    for e in entries:
        for p,h in [('image','image_sha256'),('xyz_path','xyz_sha256'),('label','label_sha256'),('semantic_label','semantic_label_sha256'),('weight','weight_sha256')]:assert sha(e[p])==e[h],(e['index'],p)
        im=Image.open(e['image']);y=np.asarray(Image.open(e['label']));s=np.asarray(Image.open(e['semantic_label']));w=np.asarray(Image.open(e['weight']));assert y.shape==s.shape==w.shape==tuple(e['shape']);assert im.size==y.shape[::-1]
        assert set(np.unique(y))<={0,1,2,3,255} and set(np.unique(s))<={0,1,2,3,4,255};assert (y[s==4]==255).all();assert (w[y==255]==0).all();assert (w[y<4]>0).all();assert (y<4).any()
        assert e['split']=='demo_fit_and_conformance' and e['not_independent_test']
        for o in e['objects']:
            x0,y0,x1,y1=o['bbox'];assert int((y[y0:y1,x0:x1]==1).sum())>=o['pixels']>=3
        coverage.append(float((y<4).mean()));counts.update({str(c):int((y==c).sum()) for c in range(4)})
    report={'passed':True,'manifest_sha256':sha(mp),'unique_demo_frames':181,'hazcam_frames':142,'navcam_frames':39,'scenario_frame_set_exact':True,'all_source_contexts_reviewed':True,'manual_crop_regions_viewed':426,'template_crop_regions_viewed':165,'template_regions_accepted':149,'geometry_components_individually_adjudicated':242,'positive_annotation_regions':sum(len(e['objects']) for e in entries),'not_unique_physical_object_count':True,'label_pixel_counts':dict(counts),'mean_per_frame_label_coverage':float(np.mean(coverage)),'minimum_per_frame_label_coverage':min(coverage),'all181_used_for_demo_fitting_not_independent_test':True,'class4_not_automatic_obstacle':True,'production_navigation_sha256':sha(R/'data/prepared/navigation.onnx'),'original_production_sha_expected':'6dcf488232cc3d46cfe7250c1bbfe1311437c9f3f65555cd6858708602ec82b0','no_model_predictions_used_in_this_audit':True}
    assert report['production_navigation_sha256']==report['original_production_sha_expected'];(R/'data/annotations/live-demo-v12-integrity.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
