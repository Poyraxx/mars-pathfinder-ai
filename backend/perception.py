"""Learned perception and strict geometry gating; no synthetic detections."""
import hashlib
import io
import json
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .data import DATA_DIR, frame_catalog, model_status
from .coordinates import project

PALETTE = np.array([[194,146,104],[112,163,176],[232,188,76],[238,91,69]],dtype=np.uint8)
_session = None
_dll_handle = None


def get_session():
    global _session, _dll_handle
    if _session is None:
        if os.name == 'nt':
            dll_dir=os.environ.get('MARS_DLL_DIR')
            config=Path(__file__).resolve().parents[1]/'.local-run.json'
            if not dll_dir and config.exists():dll_dir=json.loads(config.read_text('utf-8')).get('dll_dir')
            if dll_dir:_dll_handle=os.add_dll_directory(dll_dir)
        import onnxruntime as ort
        path = Path(os.environ.get("MARS_MODEL_PATH") or DATA_DIR / "surface.onnx")
        if not path.exists(): raise ValueError("model_not_ready")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        _session = ort.InferenceSession(str(path),sess_options=options,providers=["CPUExecutionProvider"])
    return _session


def infer(frame_id: str):
    frame = next((f for f in frame_catalog() if f["id"] == frame_id),None)
    if frame is None: raise ValueError("unknown_frame")
    source = Image.open(DATA_DIR / frame["image_path"]).convert("RGB")
    size = int(model_status().get("input_size",320))
    img = np.asarray(source.resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255
    x = ((img - [0.485,0.456,0.406]) / [0.229,0.224,0.225]).transpose(2,0,1)[None].astype(np.float32)
    session = get_session()
    began = time.perf_counter()
    logits = session.run(None,{session.get_inputs()[0].name:x})[0][0]
    logits -= logits.max(axis=0,keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=0,keepdims=True)
    mask = probs.argmax(axis=0).astype(np.uint8)
    confidence = probs.max(axis=0)
    native_mask = np.asarray(Image.fromarray(mask).resize(source.size,Image.Resampling.NEAREST))
    native_conf = np.asarray(Image.fromarray(confidence.astype(np.float32)).resize(source.size,Image.Resampling.NEAREST))
    overlay = np.asarray(source)*.48 + PALETTE[native_mask]*.52
    out = io.BytesIO()
    Image.fromarray(overlay.astype(np.uint8)).save(out,format="PNG")
    names = model_status().get("classes",["soil","bedrock","sand","big_rock"])
    result = {"frame_id":frame_id,"inference_ms":round((time.perf_counter()-began)*1000,1),
              "class_fractions":{name:round(float((native_mask==i).mean()),4) for i,name in enumerate(names)},
              "mean_class_score":round(float(native_conf.mean()),3),
              "confidence_note":"Model score, not a certified probability of correctness",
              "low_confidence_fraction":round(float((native_conf<.7).mean()),4)}
    return frame,native_mask,native_conf,out.getvalue(),result


def localize_obstacles(frame: dict, mask: np.ndarray, confidence: np.ndarray, meta: dict) -> dict:
    registration = frame.get("registration",{})
    if not registration.get("verified"):
        return {"accepted":False,"reason":"Independent orbital registration has not been verified; map unchanged."}
    error = registration.get("horizontal_error_bound_m")
    if error is None or not np.isfinite(error) or error < 0 or error > 2 or not registration.get("evidence"):
        return {"accepted":False,"reason":"Missing, unsupported or excessive registration error bound; map unchanged."}
    if not frame.get("xyz_path") or not frame.get("places"):
        return {"accepted":False,"reason":"Missing XYZ product or frame transform; map unchanged."}
    xyz = np.load(DATA_DIR / frame["xyz_path"])
    if xyz.shape != (*mask.shape,3):
        return {"accepted":False,"reason":"Reference image and XYZ geometry differ; map unchanged."}
    # Canonical label dictionary is stored in the model card, never inferred from colors.
    classes = model_status().get("classes",[])
    if "big_rock" not in classes:
        return {"accepted":False,"reason":"Model does not support big_rock."}
    valid = (mask==classes.index("big_rock")) & (confidence>=model_status().get("threshold",.7))
    valid &= np.isfinite(xyz).all(axis=-1) & (xyz!=0).any(axis=-1)
    points = xyz[valid]
    if len(points)<5:
        return {"accepted":True,"points_rc":np.empty((0,2)),"registration_error_m":error,"reason":"No geometrically supported high-score big-rock pixels."}
    places = frame["places"]
    rotation = np.array(places["archive_site_to_global_ned_rotation"],dtype=float)
    translation = np.array(places["archive_site_to_global_ned_translation_m"],dtype=float)
    if rotation.shape!=(3,3) or translation.shape!=(3,) or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or np.linalg.det(rotation)<.999:
        return {"accepted":False,"reason":"Invalid archive-to-localized frame transform; map unchanged."}
    # Corrected rover affine replaces the old telemetry anchor, avoiding ~7m drift.
    ned=points@rotation.T+translation
    lat = np.degrees(ned[:,0]/3396190.)
    lon = np.degrees(ned[:,1]/3396190.)
    xy = np.array([project(float(a),float(b),meta) for a,b in zip(lat,lon)])
    rc = np.column_stack(((meta["north_m"]-xy[:,1])/meta["cell_m"]-.5,(xy[:,0]-meta["west_m"])/meta["cell_m"]-.5))
    return {"accepted":True,"points_rc":rc,"registration_error_m":error,
            "reason":"Metric XYZ transformed through documented PLACES Site frame.","support_pixels":len(points)}
