"""Image-space non-terrain occlusion, independent of world obstacle memory."""
from functools import lru_cache
import base64
import io

import numpy as np
from PIL import Image
from scipy import ndimage

from . import data


def _fingerprint(frame):
    paths=[data.DATA_DIR/frame[k] for k in ('image_path','display_path') if frame.get(k)]
    paths += [data.DATA_DIR/name for name in ('navigation.onnx','navigation-model-card.json','surface.onnx','model-card.json')]
    return tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths if p.is_file())


@lru_cache(maxsize=160)
def _cached(frame_id, fingerprint):
    from .perception import infer_navigation
    _,labels,_,_,_=infer_navigation(frame_id)
    excluded=ndimage.binary_dilation(labels==255,iterations=1)
    rgba=np.full((*labels.shape,4),255,np.uint8);rgba[:,:,3]=np.where(excluded,0,255)
    out=io.BytesIO();Image.fromarray(rgba).save(out,format='PNG')
    return excluded,'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode('ascii')


def visibility(frame):
    if not frame.get('image_path'):
        return None,None
    return _cached(frame['id'],_fingerprint(frame))
