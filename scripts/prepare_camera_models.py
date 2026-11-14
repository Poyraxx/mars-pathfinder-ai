"""Attach real PDS CAHVORE models and measure reprojection on archived stereo XYZ.

No fitting: project measured SITE-frame points through the labelled ROVER frame
and camera model, then compare with their actual original image pixel indices.
This checks image projection, not the independent accuracy of orbital placement.
"""
from pathlib import Path
import json
import os
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import prepare_pds as P
from backend.camera_model import site_points_to_pixels


def main():
    data = ROOT/'data/prepared'
    cache = Path(os.environ.get('LOCALAPPDATA', str(Path.home())))/'mars-pathfinder/raw/pds'
    path = data/'frames.json'
    frames = json.loads(path.read_text('utf-8'))
    for f in frames:
        if f.get('instrument') != 'FRONT_HAZCAM_LEFT_A':
            continue
        lab = (cache/(f['id']+'.LBL')).read_text('ascii')
        block = P.group(lab, 'GEOMETRIC_CAMERA_MODEL_PARMS')
        rover = P.group(lab, 'ROVER_COORD_SYSTEM_PARMS')
        model = {'type': P.field(block, 'MODEL_TYPE'),
                 **{k: P.vector(block, f'MODEL_COMPONENT_{i+1}').tolist() for i,k in enumerate('CAHVORE')},
                 'T': float(P.field(block, 'MODEL_COMPONENT_8')),
                 'P': float(P.field(block, 'MODEL_COMPONENT_9')),
                 'reference_frame': P.field(block, 'REFERENCE_COORD_SYSTEM_NAME'),
                 'rover_to_site_rotation': P.rotation_wxyz(P.vector(rover,'ORIGIN_ROTATION_QUATERNION')).tolist(),
                 'rover_origin_in_site_m': P.vector(rover,'ORIGIN_OFFSET_VECTOR').tolist(),
                 'source_label_url': f['products'][1]['label_url'],
                 'source_label_sha256': f['products'][1]['label_sha256'],
                 'native_shape': f['native_shape']}
        if model['reference_frame']!='ROVER_NAV_FRAME':
            raise ValueError('Unexpected camera parent frame')
        xyz = np.load(data/f['xyz_path'])
        rows, cols = np.nonzero(np.isfinite(xyz).all(-1) & (xyz!=0).any(-1))
        uv = site_points_to_pixels(xyz[rows, cols], model)
        step = f['native_shape'][0]//f['shape'][0]
        expected = np.column_stack((cols*step+step//2, rows*step+step//2))
        errors = np.linalg.norm(uv-expected, axis=1)
        errors = errors[np.isfinite(errors)]
        valid_fraction = len(errors)/max(1,len(rows))
        p95 = float(np.percentile(errors,95)) if len(errors) else None
        model['validation'] = {'method': 'unfitted CAHVORE reprojection of archived stereo XYZ',
                               'sample_count':len(errors),'finite_fraction':valid_fraction,
                               'median_error_px':float(np.median(errors)) if len(errors) else None,
                               'p95_error_px':p95, 'accepted': p95 is not None and p95<1.5 and valid_fraction>.95}
        f['camera_model'] = model
        evidence_file=data/(f.get('registration',{}).get('evidence') or 'missing-evidence')
        if evidence_file.is_file():
            import hashlib
            evidence=json.loads(evidence_file.read_text('utf-8'))
            evidence['limitations'][0]=f"Actual {f['instrument']} image; {f['shape']} geometry grid. The camera's pixel reprojection is checked separately; this image-correlation test is about orbital placement."
            evidence_file.write_text(json.dumps(evidence,indent=2),'utf-8')
            if f['registration'].get('verified'):
                f['registration']['evidence_sha256']=hashlib.sha256(evidence_file.read_bytes()).hexdigest()
        print(f['id'],model['validation'],flush=True)
        if not model['validation']['accepted']:
            raise ValueError('Camera calibration did not reproduce source image pixels')
    path.write_text(json.dumps(frames,indent=2),'utf-8')


if __name__=='__main__':
    main()
