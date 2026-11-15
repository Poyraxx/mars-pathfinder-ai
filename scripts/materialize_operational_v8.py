"""Materialize reviewed v8/v9 polygons, preserving unknown pixels and split lineage.

Review coordinates are explicitly 512 x 512; source and masks stay native 1024.
This rasterizes human/assistant-reviewed polygons, never model-generated labels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
TRAINING = Path(os.environ['LOCALAPPDATA']) / 'mars-pathfinder/training'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', type=int, choices=[8,9], default=8)
    ap.add_argument('--trainval-only', action='store_true')
    ap.add_argument('--heldout-only', action='store_true', help='Reuse sealed train/validation entries without rewriting active training masks')
    args = ap.parse_args()
    assert not (args.trainval_only and args.heldout_only)
    version = args.version
    spec_path = ROOT / f'data/annotations/hazcam-operational-v{version}.json'
    protocol_path = ROOT / f'data/annotations/operational-v{version}-review-protocol.json'
    spec = json.loads(spec_path.read_text('utf-8'))
    protocol = json.loads(protocol_path.read_text('utf-8'))
    sources = {s['id']: s for s in json.loads((TRAINING/'manual-native/sources.json').read_text('utf-8'))}
    base = json.loads((TRAINING/f'manifest-operational-v{version-1}.json').read_text('utf-8'))
    entries = []
    for previous in base['entries']:
        row = dict(previous)
        if row['split'] in protocol['consumed_to_train']:
            row['prior_split'] = row['split']
            row['split'] = 'train'
        entries.append(row)
    if args.heldout_only:
        entries = json.loads((TRAINING/f'manifest-operational-v{version}-trainval.json').read_text('utf-8'))['entries']
    out_dir = TRAINING/f'manual-native/operational-v{version}-labels'
    out_dir.mkdir(exist_ok=True)
    review_dir = ROOT/f'artifacts/v{version}-review/labels'
    review_dir.mkdir(exist_ok=True, parents=True)
    for scene in spec['scenes']:
        if args.trainval_only and scene['split'] != 'train':
            continue
        if args.heldout_only and scene['split'] == 'train':
            continue
        source = sources[scene['id']]
        assert scene['image_sha256'] == source['image_sha256'] == sha(source['image'])
        image = Image.open(source['image']).convert('RGB')
        assert image.size == (1024, 1024)
        target = np.full((1024, 1024), 255, np.uint8)
        regions = []
        for class_id in [0, 3, 1, 2, 255]:
            for region in scene['regions']:
                if region['class_id'] != class_id:
                    continue
                p = np.asarray(region['polygon'], dtype=float)
                assert len(p) >= 3 and np.isfinite(p).all() and (p >= 0).all() and (p <= 511).all()
                p = (p*2).astype(int)
                canvas = Image.new('L', image.size)
                ImageDraw.Draw(canvas).polygon([tuple(v) for v in p], fill=1)
                target[np.asarray(canvas, bool)] = class_id
                if class_id < 4:
                    regions.append({'class_id': class_id, 'bbox': [*p.min(0).tolist(), *p.max(0).tolist()]})
        # Exclude uncertain native boundary pixels, including the review ROI edge.
        boundary = np.zeros(target.shape, bool)
        for axis in [0, 1]:
            change = np.diff(target.astype(np.int16), axis=axis) != 0
            a, b = [slice(None)]*2, [slice(None)]*2
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            boundary[tuple(a)] |= change
            boundary[tuple(b)] |= change
        target[ndimage.binary_dilation(boundary, iterations=6)] = 255
        output = out_dir/(scene['id']+'.png')
        Image.fromarray(target).save(output)
        counts = {str(c): int((target == c).sum()) for c in range(4)}
        row = {'source_id': scene['id'], 'image': source['image'], 'label': str(output),
               'group': f"manual-sol-{scene['sol']}", 'split': scene['split'], 'sol': scene['sol'],
               'source_sha256': source['image_sha256'], 'label_sha256': sha(output),
               'pixel_counts': counts, 'regions': regions, 'annotation': f'operational-v{version} visually traced polygons'}
        entries.append(row)
        rgb = np.asarray(image.resize((512, 512))).astype(float)
        small = np.asarray(Image.fromarray(target).resize((512, 512), Image.Resampling.NEAREST))
        palette = np.array([[50, 180, 120], [240, 70, 50], [60, 155, 245], [200, 180, 80]])
        for c in range(4):
            rgb[small == c] = .55*rgb[small == c] + .45*palette[c]
        Image.fromarray(rgb.clip(0, 255).astype('uint8')).save(review_dir/f"sol{scene['sol']}.jpg", quality=96)
    assert len({e['source_id'] for e in entries}) == len(entries)
    train = [e for e in entries if e['split'] == 'train']
    held = [e for e in entries if e['split'] != 'train']
    assert not ({e['source_sha256'] for e in train} & {e['source_sha256'] for e in held})
    assert not ({e['group'] for e in train} & {e['group'] for e in held})
    for e in entries:
        if e['split'] != f'test_v{version}':
            continue
        assert min(abs(e['sol']-t['sol']) for t in train) > 6
        rmc = re.search(r'RAD_F(\d{3})(\d{4})', e['source_id'])
        assert rmc and not any(re.search(r'RAD_F(\d{3})(\d{4})', t['source_id']).groups() == rmc.groups() for t in train)
    manifest = {**{k:v for k,v in base.items() if k != 'entries'}, 'version': version,
                'protocol': f'V{version} protocol and reviewed polygon hashes; consumed earlier tests explicitly training, fixed validation unchanged.',
                'review_protocol_sha256': sha(protocol_path), 'annotation_sha256': sha(spec_path), 'entries': entries}
    name = f'manifest-operational-v{version}-trainval.json' if args.trainval_only else f'manifest-operational-v{version}.json'
    path = TRAINING/name
    path.write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    print(json.dumps({'manifest':str(path), 'sha256':sha(path), 'counts':dict(Counter(e['split'] for e in entries)),
                      'new_scenes':len(entries)-len(base['entries'])}), flush=True)


if __name__ == '__main__':
    main()
