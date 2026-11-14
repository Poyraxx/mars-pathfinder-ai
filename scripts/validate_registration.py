"""Validate independently selected camera/orbital control pairs before enabling fusion.

Input: {frame_id, source: 'independent_manual_controls', controls: [
  {camera_pixel: [column,row], orbital_crop_pixel: [column,row]} ... ]}

The controls MUST identify the same physical feature independently in both real
images. No tie points are synthesized, inferred from the same frame transform, or
provided with this project. This tool measures the archived transform; it does not
fit the transform to the controls and then score its own fitted points.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'prepared'


def validate(payload,frame,orbital,xyz):
    if payload.get('source')!='independent_manual_controls' or len(payload.get('controls',[]))<6:
        raise ValueError('At least six independently identified feature pairs are required')
    rotation=np.array(frame['places']['archive_site_to_global_ned_rotation'])
    translation=np.array(frame['places']['archive_site_to_global_ned_translation_m'])
    residuals=[];used=set()
    for control in payload['controls']:
        x,y=control['camera_pixel'];u,v=control['orbital_crop_pixel']
        if (x,y) in used:raise ValueError('Duplicate camera controls are not independent')
        used.add((x,y))
        if not(0<=x<xyz.shape[1] and 0<=y<xyz.shape[0] and 0<=u<256 and 0<=v<256):raise ValueError('Control outside reference imagery')
        point=xyz[int(y),int(x)]
        if not np.isfinite(point).all() or not point.any():raise ValueError('Control has no valid stereo point')
        ned=rotation@point+translation
        expected_e=orbital['upper_left_easting_m']+(u+.5)*orbital['pixel_resolution_m']
        expected_n=orbital['upper_left_northing_m']-(v+.5)*orbital['pixel_resolution_m']
        error=float(np.hypot(ned[1]-expected_e,ned[0]-expected_n))
        residuals.append({'camera_pixel':[x,y],'orbital_crop_pixel':[u,v],'horizontal_residual_m':error})
    values=np.array([r['horizontal_residual_m'] for r in residuals])
    return {'verified':bool(values.max()<=2),'horizontal_error_bound_m':float(values.max()),
            'rmse_m':float(np.sqrt((values**2).mean())),'control_count':len(values),
            'method':'Independent manually identified controls; empirical max residual, not a probabilistic confidence interval',
            'controls':residuals}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('controls',type=Path);args=parser.parse_args()
    payload=json.loads(args.controls.read_text('utf-8'));frames=json.loads((DATA/'frames.json').read_text('utf-8'))
    frame=next(f for f in frames if f['id']==payload['frame_id'])
    orbital=json.loads((DATA/'pds'/'orbital-crop.json').read_text('utf-8'))
    result=validate(payload,frame,orbital,np.load(DATA/frame['xyz_path']))
    evidence_path=DATA/'pds'/frame['id']/'registration-evidence.json'
    result['source_controls_sha256']=hashlib.file_digest(args.controls.open('rb'),'sha256').hexdigest()
    result['camera_product_id']=frame['reference_image_product_id'];result['orbital_source']=orbital
    evidence_path.write_text(json.dumps(result,indent=2),'utf-8')
    frame['registration']={**result,'evidence':str(evidence_path.relative_to(DATA)).replace('\\','/')}
    (DATA/'frames.json').write_text(json.dumps(frames,indent=2),'utf-8')
    print(json.dumps(result,indent=2))
    if not result['verified']:raise SystemExit('Registration failed; map fusion remains disabled')


if __name__=='__main__':main()
