"""Check native reviewed inputs and split independence without model predictions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from PIL import Image
from scipy import ndimage


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--terrain-manifest', type=Path)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--annotations', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text('utf8'))
    assert manifest['review_protocol_sha256'] == sha(args.protocol)
    assert manifest['annotation_sha256'] == sha(args.annotations)
    counts, pixels, objects = Counter(), Counter(), Counter()
    unique_ids = set()
    partitions = {name: {} for name in ('source', 'group', 'rmc')}
    terrain_groups = None
    if args.terrain_manifest:
        terrain = json.loads(args.terrain_manifest.read_text('utf8'))
        terrain_groups = {e['group'] for e in terrain['entries'] if e['split'] == 'train'}
    for entry in manifest['entries']:
        split = entry['split']
        assert entry['source_id'] not in unique_ids
        unique_ids.add(entry['source_id'])
        assert sha(entry['image']) == entry['source_sha256']
        assert sha(entry['label']) == entry['label_sha256']
        source = Image.open(entry['image'])
        label = np.asarray(Image.open(entry['label']))
        assert source.size == (1024, 1024) and label.shape == (1024, 1024)
        assert set(np.unique(label)) <= {0, 1, 2, 3, 255}
        recorded = entry['pixel_counts']
        if isinstance(recorded, list):
            recorded = {str(c): value for c, value in enumerate(recorded)}
        assert {str(c): int((label == c).sum()) for c in range(4)} == recorded
        rmc = re.search(r'RAD_F(\d{3})(\d{4})', entry['source_id'])
        assert rmc, entry['source_id']
        for name, value in [('source', entry['source_sha256']), ('group', entry['group']), ('rmc', rmc.groups())]:
            previous = partitions[name].setdefault(value, split)
            assert previous == split, (name, value, previous, split)
        if split == f"test_v{manifest['version']}":
            assert min(abs(entry['sol']-other['sol']) for other in manifest['entries'] if other['split'] == 'train') > 6
            if terrain_groups is not None:
                clock = int(re.search(r'_(\d{9})', entry['source_id'])[1])
                group = clock // (88775*3)
                assert not ({str(group+d) for d in (-1, 0, 1)} & terrain_groups)
        counts[split] += 1
        pixels[split] += int((label < 4).sum())
        components, _ = ndimage.label(label == 1)
        objects[split] += int((np.bincount(components.ravel())[1:] >= 50).sum())
    report = {'images': len(unique_ids), 'split_counts': dict(counts),
              'reviewed_pixels': dict(pixels), 'scored_connected_objects_min_50px': dict(objects),
              'all_source_label_hashes_match': True, 'same_rmc_cross_split_collisions': 0,
              'manifest_sha256': sha(args.manifest),
              'annotation_sha256': sha(args.annotations), 'review_protocol_sha256': sha(args.protocol),
              'frozen_terrain_train_buffer_checked': terrain_groups is not None,
              'limitations': 'Sparse visually reviewed regions; no assertion of independent expert truth or exact physical rock dimensions.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
