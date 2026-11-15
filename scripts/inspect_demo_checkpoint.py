"""Source/actual trained-model overlay for offline review, never relabeling."""
import argparse,json
from pathlib import Path
from train_live_demo_v12 import DemoNavigation,normalize,torch,sha
from PIL import Image,ImageDraw
import numpy as np
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--init',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--indices',type=int,nargs='+',required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);base=DemoNavigation(a.init).eval();candidate=DemoNavigation(a.init).eval();candidate.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True));entries=json.loads(a.manifest.read_text())['entries'];records=[]
    for start in range(0,len(a.indices),3):
        sheet=Image.new('RGB',(1536,1632),'#101820');draw=ImageDraw.Draw(sheet)
        for row,index in enumerate(a.indices[start:start+3]):
            e=entries[index];source=Image.open(e['image']).convert('RGB');rgb=np.asarray(source.resize((512,512)));sheet.paste(Image.fromarray(rgb),(0,row*544+28));draw.text((4,row*544+4),f"{index:03} source / original / trained checkpoint (red>=.7 rock; blue>=.8 hardware)",fill='white');rec={'index':index,'source_sha256':sha(e['image'])}
            for column,net in [(1,base),(2,candidate)]:
                with torch.no_grad():p=net(normalize(source)[None]).softmax(1)[0].numpy()
                labels=p.argmax(0);rock=(labels==1)&(p[1]>=.7);hw=(labels==2)&(p[2]>=.8);marked=rgb.copy();marked[rock]=(.55*marked[rock]+.45*np.array([250,60,50])).astype('uint8');marked[hw]=(.55*marked[hw]+.45*np.array([50,145,250])).astype('uint8');sheet.paste(Image.fromarray(marked),(column*512,row*544+28));rec['original' if column==1 else 'trained']={'rock_pixels512':int(rock.sum()),'hardware_pixels512':int(hw.sum())}
            records.append(rec)
        sheet.save(a.output/f'page-{start//3:02}.jpg',quality=95)
    (a.output/'report.json').write_text(json.dumps({'checkpoint_sha256':sha(a.checkpoint),'manifest_sha256':sha(a.manifest),'scope':'Offline actual checkpoint outputs, no manual labels substituted and no inference policy changes.','frames':records},indent=2),encoding='utf8')
if __name__=='__main__':main()
