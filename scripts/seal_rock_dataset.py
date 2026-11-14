"""Freeze group-buffered expansion and an untouched official expert benchmark."""
import hashlib
import gzip
import json
import os
import re
from collections import Counter
from pathlib import Path


def neighbours(groups):
    return {str(int(g)+delta) for g in groups for delta in (-1,0,1)}


def main():
    root=Path(os.environ['LOCALAPPDATA'])/'mars-pathfinder/training'
    baseline=json.loads((root/'v3-audit-baseline/manifest.json').read_text('utf8'))
    expansion=json.loads((root/'manifest-v4-expanded.json').read_text('utf8'))
    sealed=json.loads((root/'manifest-sealed-gold.json').read_text('utf8'))
    manual=json.loads((root/'manifest-manual-native.json').read_text('utf8'))
    old_groups={e['group'] for e in baseline['entries']}
    manual_clock_group=lambda e:str(int(re.search(r'_(\d{9})',e['source_id'])[1])//(88775*3))
    manual_train_groups={manual_clock_group(e) for e in manual['entries'] if e['split']=='train'}
    manual_heldout_groups={manual_clock_group(e) for e in manual['entries'] if e['split']!='train'}
    gold=[e for e in sealed['entries'] if e['group'] not in neighbours(old_groups|manual_train_groups)]
    assert gold, 'No independent expert groups remain'
    forbidden=neighbours({e['group'] for e in baseline['entries'] if e['split']!='train'}|{e['group'] for e in gold}|manual_heldout_groups)
    entries=[e for e in expansion['entries'] if e['split']!='train' or e['group'] not in forbidden]
    # Manual validation/test stay separate: a tiny sparse check must not distort
    # or replace the large, fixed validation set used for checkpoint selection.
    entries += [e for e in manual['entries'] if e['split']=='train']
    protocol='Frozen original validation/test; new train excludes their +/-1 three-sol groups, sealed expert groups +/-1, and manual heldout camera episodes +/-1 across sensors. Sealed expert excludes ALL v3 train/val/test and manual training groups +/-1. Sparse native Hazcam training only; its val/test are separate.'
    final={**expansion,'entries':entries,'split_method':protocol,
           'reserved_expert_groups':sorted({e['group'] for e in gold}),
           'counts':dict(Counter(e['split'] for e in entries))}
    sealed={**sealed,'entries':gold,'protocol':protocol,'opened_for_model_selection':False}
    output=root/'manifest-v4-final.json';output.write_text(json.dumps(final,indent=2),'utf8')
    (root/'manifest-sealed-gold-final.json').write_text(json.dumps(sealed,indent=2),'utf8')
    audit={'protocol':protocol,'split_counts':final['counts'],'train_rock_scenes':sum(bool(e.get('rock_boxes')) for e in entries if e['split']=='train'),
           'sealed_gold_images':len(gold),'sealed_gold_groups':len({e['group'] for e in gold}),
           'manifest_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
    (root/'dataset-v4-audit.json').write_text(json.dumps(audit,indent=2),'utf8');print(json.dumps(audit),flush=True)
    portable={'source':baseline['source'],'license':baseline['license'],'protocol':protocol,
              'baseline_v3':[{k:v for k,v in e.items() if k not in ('image','label')} for e in baseline['entries']],
              'entries':[{k:v for k,v in e.items() if k not in ('image','label')} for e in entries],
              'sealed_gold':[{k:v for k,v in e.items() if k not in ('image','label')} for e in gold]}
    target=Path(__file__).resolve().parents[1]/'data/annotations/dataset-v4-splits.json.gz'
    with gzip.GzipFile(filename=str(target),mode='wb',mtime=0) as stream:
        stream.write(json.dumps(portable,sort_keys=True,separators=(',',':')).encode('utf8'))


if __name__=='__main__':main()
