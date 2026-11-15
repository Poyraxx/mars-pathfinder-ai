"""Render native-pixel crops around visually placed sparse stone interiors."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training'
P=ROOT/'data/annotations/operational-v10-visual-review.json';OUT=ROOT/'artifacts/v10-review/native-crops'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--accepted-only',action='store_true');ap.add_argument('--ground-only',action='store_true');args=ap.parse_args();assert not (args.accepted_only and args.ground_only)
    spec=json.loads(P.read_text('utf8'));entries={e['source_id']:e for e in json.loads((T/'manifest-operational-v9.json').read_text('utf8'))['entries']}
    output=ROOT/'artifacts/v10-review/accepted-native' if args.accepted_only else OUT
    accepted=None;adjustments={}
    if args.accepted_only:
        qa=json.loads((ROOT/'data/annotations/operational-v10-native-qa.json').read_text('utf8'))
        accepted={key for p in qa['pages'] for key in p['accepted']};adjustments={k:v for p in qa['pages'] for k,v in p['manual_adjustments'].items()}
    if args.ground_only:
        ground=json.loads((ROOT/'data/annotations/operational-v10-ground-review.json').read_text('utf8'))
        accepted={g['proposal_id'] for g in ground['interiors']};adjustments={g['proposal_id']:g['interior_ellipse_512'] for g in ground['interiors']};output=ROOT/'artifacts/v10-review/ground-native'
    output.mkdir(parents=True,exist_ok=True);rows=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
    for s in spec['scenes']:
        e=entries[s['source_id']];im=Image.open(e['image']).convert('RGB');old=np.asarray(Image.open(e['label']))
        assert hashlib.sha256(Path(e['image']).read_bytes()).hexdigest()==s['source_sha256']
        for index,(x,y,rx,ry) in enumerate(s['small_rock_interiors']):
            key=f'{s["index"]}.{index}'
            if accepted is not None and key not in accepted:continue
            x,y,rx,ry=adjustments.get(key,(x,y,rx,ry))
            box=(x*2-32,y*2-32,x*2+32,y*2+32);crop=im.crop(box);a=np.asarray(crop)
            if np.median(a)<75:crop=Image.fromarray((255*np.power(a.astype(float)/255,.45)).astype('uint8'))
            crop=crop.resize((160,160),Image.Resampling.NEAREST);draw=ImageDraw.Draw(crop)
            centre=(80,80);draw.ellipse((80-rx*5,80-ry*5,80+rx*5,80+ry*5),outline='#ff51e7',width=1)
            prior=int(old[y*2,x*2]);rows.append({'id':f'{s["index"]}.{index}','scene_index':s['index'],'seed_index':index,'sol':s['sol'],'crop':crop,'old_class':prior})
    for page in range((len(rows)+39)//40):
        canvas=Image.new('RGB',(1440,1000),'#101820');d=ImageDraw.Draw(canvas)
        for pos,r in enumerate(rows[page*40:page*40+40]):
            x=pos%8*180;y=pos//8*200
            d.text((x+5,y+2),f"{r['id']} / Sol{r['sol']} / old{r['old_class']}",font=font,fill='white')
            canvas.paste(r['crop'],(x+10,y+25))
        canvas.save(output/f'qa-{page:02}.jpg',quality=97)
    (output/'index.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='crop'} for r in rows],indent=2),encoding='utf8')
    print(len(rows),'proposed interiors in',(len(rows)+39)//40,'native QA pages')
if __name__=='__main__':main()
