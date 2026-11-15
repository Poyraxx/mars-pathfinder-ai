"""Download archive-only candidates; eligibility uses source groups, never predictions."""
import concurrent.futures,hashlib,json,os,re
from pathlib import Path
import httpx,numpy as np
from PIL import Image
import prepare_pds as P
ROOT=Path(__file__).resolve().parents[1];T=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training';N=T/'manual-native'
def main():
    portable=ROOT/'data/annotations/operational-v11-source-pool.json'
    expected={r['sol']:r for r in json.loads(portable.read_text('utf8'))['sources'] if 'error' not in r} if portable.exists() else {}
    pool=sorted(expected) if expected else json.loads((ROOT/'artifacts/v11-pds-pool.json').read_text('utf8'))['selected_sols']
    (N/'raw').mkdir(exist_ok=True,parents=True);(N/'images').mkdir(exist_ok=True,parents=True)
    old=json.loads((T/'manifest-operational-v9.json').read_text('utf8'))['entries'];rmcs={re.search(r'RAD_F(\d{3})(\d{4})',e['source_id']).groups() for e in old}
    terrain=json.loads((T/'manifest-v4-final.json').read_text('utf8'));groups={e['group'] for e in terrain['entries'] if e['split']=='train' and e['group'].isdigit()}
    def fetch(sol):
        try:
            base=f'https://planetarydata.jpl.nasa.gov/img/data/msl/MSLHAZ_1XXX/DATA/SOL{sol:05d}/';r=httpx.get(base,timeout=45);r.raise_for_status()
            ids=[expected[sol]['id']] if sol in expected else sorted(set(re.findall(r'href="(FL[AB]_\d+RAD_F[^"/]+)\.LBL"',r.text)));assert ids,'No native forward RAD'
            sid=ids[0];lb=N/'raw'/(sid+'.LBL');ip=N/'raw'/(sid+'.IMG');label=P.fetch(base+sid+'.LBL',lb);binary=P.fetch(base+sid+'.IMG',ip)
            values,_=P.decode_pds(label,binary);a=values[0].astype(float);assert a.shape==(1024,1024)
            finite=a[np.isfinite(a)&(a>0)];lo,hi=np.percentile(finite,[1,99]);image=Image.fromarray(np.uint8(np.clip((a-lo)/(hi-lo),0,1)*255)).convert('RGB');path=N/'images'/(sid+'.png')
            if not path.exists():image.save(path)
            assert np.array_equal(np.asarray(Image.open(path).convert('RGB')),np.asarray(image)),'Existing prepared source differs from the raw RAD'
            if sol in expected:
                assert hashlib.sha256(binary).hexdigest()==expected[sol]['raw_sha256']
                assert hashlib.sha256(path.read_bytes()).hexdigest()==expected[sol]['image_sha256']
            clock=int(re.search(r'_(\d{9})',sid)[1]);group=clock//(88775*3);rmc=re.search(r'RAD_F(\d{3})(\d{4})',sid).groups()
            return {'id':sid,'sol':sol,'image':str(path),'shape':[1024,1024],'source_url':base+sid+'.IMG','label_url':base+sid+'.LBL','raw_sha256':hashlib.sha256(binary).hexdigest(),'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'preparation':'Actual PDS RAD percentile1-99 stretch; no synthetic content','clock_group':str(group),'rmc':list(rmc),'terrain_training_buffer_overlap':bool({str(group+d) for d in [-1,0,1]}&groups),'old_custom_rmc_collision':rmc in rmcs,'source_selected_without_model_predictions':True}
        except Exception as error:return {'sol':sol,'error':str(error)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:rows=list(ex.map(fetch,pool))
    path=N/'v11-new-sources.json';path.write_text(json.dumps(rows,indent=2),encoding='utf8')
    print(json.dumps([{k:r[k] for k in ['sol','error','rmc','terrain_training_buffer_overlap','old_custom_rmc_collision'] if k in r} for r in rows]),flush=True)
if __name__=='__main__':main()
