"""Build a NEW five-class sparse review without rewriting historical masks."""
import gzip,hashlib,json,os
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage
ROOT=Path(__file__).resolve().parents[1];T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf8')
def main():
    review_path=ROOT/'data/annotations/operational-v10-visual-review.json';qa_path=ROOT/'data/annotations/operational-v10-native-qa.json'
    review=json.loads(review_path.read_text('utf8'));qa=json.loads(qa_path.read_text('utf8'))
    assert len(review['scenes'])==86 and [p['page'] for p in qa['pages']]==list(range(15))
    approved={sid for p in qa['pages'] for sid in p['accepted']};adjustments={k:v for p in qa['pages'] for k,v in p['manual_adjustments'].items()}
    ground_path=ROOT/'data/annotations/operational-v10-ground-review.json';ground=json.loads(ground_path.read_text('utf8'))
    base_path=T/'manifest-operational-v9.json';base=json.loads(base_path.read_text('utf8'));assert sha(base_path)==review['source_manifest_sha256']
    by_id={e['source_id']:e for e in base['entries']};out=T/'manual-native/operational-v10-labels';out.mkdir(exist_ok=True)
    scenes=[];entries=[];stats=Counter();palette=np.array([[50,165,110],[240,55,45],[50,145,240],[180,155,65],[225,95,225]])
    preview=ROOT/'artifacts/v10-review/final';preview.mkdir(exist_ok=True,parents=True)
    for s in review['scenes']:
        e=by_id[s['source_id']];assert sha(e['image'])==e['source_sha256'] and sha(e['label'])==e['label_sha256']
        old=np.asarray(Image.open(e['label']));target=old.copy();accepted=[];cores=np.zeros(old.shape,bool);uncertain=np.zeros(old.shape,bool)
        for i,seed in enumerate(s['small_rock_interiors']):
            key=f'{s["index"]}.{i}'
            if key not in approved:continue
            x,y,rx,ry=adjustments.get(key,seed);p=Image.new('L',(1024,1024));d=ImageDraw.Draw(p)
            d.ellipse((2*(x-rx),2*(y-ry),2*(x+rx),2*(y+ry)),fill=1);core=np.asarray(p,bool)
            writable=~np.isin(old,[1,2,3]);core &= writable
            if core.sum()<8:continue
            # Interior supervision only. Do not call the untraced remainder of
            # a stone surrounding its reviewed core negative ground.
            yy,xx=np.ogrid[:1024,:1024];roi=(np.abs(xx-2*x)<=32)&(np.abs(yy-2*y)<=32)
            uncertain |= roi & writable;cores |= core
            accepted.append({'proposal_id':key,'interior_ellipse_512':[x,y,rx,ry],'uncertain_roi_native':[max(0,2*x-32),max(0,2*y-32),min(1023,2*x+32),min(1023,2*y+32)],'reviewed_core_pixels':int(core.sum())})
        target[uncertain]=255;target[cores]=4
        # Old0 means ground OR small stone, not a certain bare-ground negative.
        # A separate bitmask preserves this partial label during training.
        known_ground=np.zeros(old.shape,bool);negative_regions=[]
        for negative in ground['interiors']:
            if negative['source_id']!=s['source_id']:continue
            x,y,rx,ry=negative['interior_ellipse_512'];p=Image.new('L',(1024,1024))
            ImageDraw.Draw(p).ellipse((2*(x-rx),2*(y-ry),2*(x+rx),2*(y+ry)),fill=1)
            core=np.asarray(p,bool)&~np.isin(old,[1,2,3]);assert not (core&cores).any()
            known_ground |= core
            negative_regions.append({'class_id':0,'reviewed_bare_ground':True,'bbox':[2*(x-rx),2*(y-ry),2*(x+rx),2*(y+ry)]})
        target[known_ground]=0;partial=(target==0)&~known_ground
        assert np.array_equal(target==1,old==1) and np.array_equal(target==2,old==2) and np.array_equal(target==3,old==3)
        assert set(np.unique(target))<={0,1,2,3,4,255}
        path=out/(s['source_id']+'.png');Image.fromarray(target).save(path)
        partial_path=out/(s['source_id']+'-partial.png');Image.fromarray(partial.astype('uint8')).save(partial_path)
        role='consumed_control' if e['split']=='test_v9' else e['split']
        row={**e,'label':str(path),'prior_label_sha256':e['label_sha256'],'label_sha256':sha(path),'split':role,'prior_split':e['split'],'pixel_counts':{str(c):int((target==c).sum()) for c in range(5)},'annotation':'v10 visually checked sparse small-rock interiors; historical1/2/3 masks preserved'}
        row.update(partial_ground_mask=str(partial_path),partial_ground_sha256=sha(partial_path),partial_ground_pixels=int(partial.sum()),known_bare_ground_pixels=int(known_ground.sum()))
        new_regions=[{'class_id':4,'bbox':[int(2*(x-rx)),int(2*(y-ry)),int(2*(x+rx)),int(2*(y+ry))]} for a in accepted for x,y,rx,ry in [a['interior_ellipse_512']]]
        row['regions']=[*e['regions'],*new_regions,*negative_regions]
        entries.append(row);stats['accepted_interiors']+=len(accepted);stats['small_rock_pixels']+=int((target==4).sum());stats['ground_pixels_moved_to_unknown']+=int(((old==0)&(target==255)).sum());stats['old_unknown_pixels_now_rock']+=int(((old==255)&(target==4)).sum())
        stats['known_bare_ground_pixels']+=int(known_ground.sum())
        scenes.append({'source_id':s['source_id'],'sol':s['sol'],'source_sha256':s['source_sha256'],'reviewed_full_source':True,'native_crop_qa_completed':True,'initial_proposals':len(s['small_rock_interiors']),'accepted':accepted,'prior_mask_sha256':e['label_sha256'],'mask_sha256':sha(path),'original_large_obstacle_hardware_sky_masks_unchanged':True})
        im=np.asarray(Image.open(e['image']).convert('RGB')).astype(float)
        for c in range(5):im[target==c]=im[target==c]*.6+palette[c]*.4
        view=Image.fromarray(im.clip(0,255).astype('uint8'));view.thumbnail((512,512));view.save(preview/f'sol{s["sol"]}.jpg',quality=96)
    protocol={'version':10,'production_model_unchanged':True,'baseline_navigation_sha256':'6dcf488232cc3d46cfe7250c1bbfe1311437c9f3f65555cd6858708602ec82b0','source_images_and_old_masks_immutable':True,'classes':review['classes'],'reviewed_all_custom_operational_sources':86,'proposed_interiors':sum(len(s['small_rock_interiors']) for s in review['scenes']),'accepted_interiors':stats['accepted_interiors'],'method':'Sparse visually checked interior ellipses, not exact silhouettes; untraced64-native-pixel context becomes unknown, original obstacle/hardware/sky pixels always protected. Preview brightness clipping was discarded before native QA; original source bytes never altered.','physical_size':'Class4 is small-rock appearance in these images, not measured dimensions or proof of passability. Stereo geometry still required for occupancy.','training':'73 train,5 validation,8 previously opened consumed controls; five-class head initialized from v11, frozen original terrain branch. Select on validation only.','training_plan':{'size':512,'epochs':6,'samples_per_epoch':1008,'head_lr':1e-5,'encoder_lr_factor':.1,'ground_error_weight':1.25,'obstacle_error_weight':.75,'selection':'0.8*unchanged four-class collapsed operational objective +0.2*small-rock IoU; report both separately.','deployment':'Research candidate only. Keep v11 if any large-object regression or original task metric drops. No new independent test exists in this revision; no automatic deployment.'},'expert_datasets':'Original AI4Mars/HiRISE publisher labels and terrain taxonomy are preserved. This86-source sweep is not falsely described as re-labelling thousands of external expert/crowd-labelled frames.'}
    protocol['partial_labels']='Historical0 permits{0,4}: coarse ground/small-stone labels do not become false negatives for untraced small rocks.63 visually checked bare-ground cores provide explicit0 supervision. New4 is a checked rock interior. Partial0 pixels are excluded from strict five-class metrics but included as0 in the old four-class collapsed task.'
    protocol['ground_review_sha256']=sha(ground_path)
    protocol['mask_composition']='Union all accepted cores before applying uncertain ROI; nearby stone interiors cannot erase one another.'
    protocol['interrupted_pilot']='v19 stopped after detecting nearby-core composition and coarse-ground label ambiguity, before any control-set evaluation; retained locally as interrupted, not a completed release.'
    protocol['training_plan'].update(head_lr=1e-5,partial_ground_loss='-log(p_ground+p_small_rock), with separate hard labels for reviewed ground and rock cores',small_rock_class_weight=16)
    protocol_path=ROOT/'data/annotations/operational-v10-protocol.json';save(protocol_path,protocol)
    manifest={'version':10,'classes':review['classes'],'source_manifest_sha256':sha(base_path),'review_sha256':sha(review_path),'native_qa_sha256':sha(qa_path),'protocol_sha256':sha(protocol_path),'entries':entries}
    mp=T/'manifest-operational-v10.json';save(mp,manifest)
    save(ROOT/'data/annotations/operational-v10-revision.json',{'stats':dict(stats),'scenes':scenes,'manifest_sha256':sha(mp),'review_sha256':sha(review_path),'native_qa_sha256':sha(qa_path),'protocol_sha256':sha(protocol_path)})
    portable={**manifest,'entries':[{k:v for k,v in e.items() if k not in ('image','label','partial_ground_mask')} for e in entries]}
    with gzip.GzipFile(str(ROOT/'data/annotations/operational-v10-splits.json.gz'),'wb',mtime=0) as f:f.write(json.dumps(portable,indent=1).encode())
    print(json.dumps({'manifest':str(mp),'sha256':sha(mp),'roles':dict(Counter(e['split'] for e in entries)),**dict(stats)}),flush=True)
if __name__=='__main__':main()
