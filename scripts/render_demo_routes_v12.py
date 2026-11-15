"""QA images from the actual measured/projected route outputs, no fitted SVG."""
import argparse,gzip,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--details',type=Path,required=True);ap.add_argument('--comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    details=json.loads(gzip.decompress(a.details.read_bytes()));comparison=json.loads(a.comparison.read_text('utf8'));inventory=json.loads((R/'data/annotations/live-demo-v12-inventory.json').read_text('utf8'))['entries'];byid={e['frame_id']:e for e in inventory};catalog={f['id']:f for f in json.loads((R/'data/prepared/frames.json').read_text('utf8'))};font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    selected=list(dict.fromkeys((c['scenario'],c['frame_id']) for c in comparison['changed_frame_scenario_entries']))
    sid='hazcam-extended-traverse'
    selected+= [(sid,e['frame_id']) for e in inventory if e['index'] in [3,5,105,123,129,131,132,138,141] and (sid,e['frame_id']) not in selected]
    for page in range((len(selected)+3)//4):
        canvas=Image.new('RGB',(1024,1200),'#101820');d=ImageDraw.Draw(canvas)
        for n,(sid,fid) in enumerate(selected[page*4:page*4+4]):
            row=details[sid]['frames'][fid];f=catalog[fid];im=Image.open(R/'data/prepared'/(f.get('display_path') or f['image_path'])).convert('RGB').resize((512,512));overlay=Image.new('RGBA',(512,512));draw=ImageDraw.Draw(overlay)
            for poly in row.get('ribbons',[]):draw.polygon([(x*512,y*512) for x,y in poly],fill=(40,220,245,100))
            for line in row.get('segments',[]):draw.line([(p[0]*512,p[1]*512) for p in line],fill=(90,240,255,235),width=2)
            for kind,color in [('obstacles',(255,60,50,220)),('tracked_obstacles',(255,170,40,220))]:
                for x,y in row.get(kind,[]):draw.ellipse((x*512-1,y*512-1,x*512+1,y*512+1),fill=color)
            im=Image.alpha_composite(im.convert('RGBA'),overlay).convert('RGB');x=(n%2)*512;y=(n//2)*600;canvas.paste(im,(x,y+54));d.text((x+5,y+3),f"{byid[fid]['index']:03} {row['status']} / {sid}",font=font,fill='white');d.text((x+5,y+25),f"{row.get('failure_kind','')} | measured route only",font=font,fill='#9cb4c2');reason=row.get('reason','');d.text((x+5,y+569),reason[:64],font=font,fill='white')
        canvas.save(a.output/f'route-qa-{page:02}.jpg',quality=94)
    (a.output/'index.json').write_text(json.dumps({'candidate_sha256':comparison['candidate_sha256'],'frames':[{'scenario':s,'frame_id':f,'index':byid[f]['index']} for s,f in selected],'pages':(len(selected)+3)//4},indent=2),encoding='utf8');print('Rendered',len(selected),'actual route contexts',flush=True)
if __name__=='__main__':main()
