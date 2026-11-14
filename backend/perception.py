"""Learned perception and strict geometry gating; no synthetic detections."""
import hashlib
import io
import json
import os
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
from .data import DATA_DIR, frame_catalog, model_status
from .coordinates import project

PALETTE = np.array([[194,146,104],[112,163,176],[232,188,76],[238,91,69]],dtype=np.uint8)
_session = None
_dll_handle = None


@lru_cache(maxsize=128)
def _file_sha(path: str, size: int, mtime_ns: int):
    # Size/mtime participate in the key; changed local inputs invalidate the cache.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha(path: Path):
    st=path.stat()
    return _file_sha(str(path),st.st_size,st.st_mtime_ns)


def registration_evidence_matches(frame: dict) -> bool:
    """A passing flag cannot outlive its evidence or exact camera/XYZ inputs."""
    reg=frame.get('registration',{})
    try:
        path=DATA_DIR/reg['evidence']
        if _sha(path)!=reg['evidence_sha256']:
            return False
        proof=json.loads(path.read_text('utf-8'))
        if not proof.get('verified') or proof.get('frame_id')!=frame['id']:
            return False
        if abs(proof['horizontal_error_bound_m']-reg['horizontal_error_bound_m'])>1e-8:
            return False
        hashes=proof['inputs_sha256']
        return _sha(DATA_DIR/frame['xyz_path'])==hashes['xyz_npy'] and _sha(DATA_DIR/frame['image_path'])==hashes['reference_png']
    except (KeyError,OSError,TypeError,ValueError):
        return False


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


STEREO_CELL_M = 0.1
GROUND_CELL_M = 0.25
GROUND_WINDOW_M = 2.0
GROUND_PERCENTILE = 20
MAX_STEREO_ERROR_M = 0.10
MIN_COMPONENT_PX = 6
MAX_FOOTPRINT_M = 3.0  # discrete-obstacle detector: wider raised areas are terrain shape or stereo artefacts
MAX_FOOTPRINT_AREA_M2 = 2.0


def stereo_hazards(xyz: np.ndarray, xye: np.ndarray, step_m: float) -> dict:
    """Measured geometric hazards: pixels standing >= step_m above local stereo ground.

    xyz is SITE-frame north/east/down in metres. Local ground is the 20th percentile of
    per-cell minimum heights within a 2 m window. Only pixels whose XYE range-error norm is
    <= 0.10 m are used. Components wider than MAX_FOOTPRINT_M on the ground are dropped:
    this is a discrete-obstacle (rock) detector, not a slope or ledge detector. No learned
    class is involved.
    """
    from scipy import ndimage
    valid = np.isfinite(xyz).all(-1) & (xyz != 0).any(-1)
    err = np.linalg.norm(np.nan_to_num(xye, nan=np.inf), axis=-1) if xye is not None else np.full(xyz.shape[:2], np.inf)
    empty = {"mask": np.zeros(xyz.shape[:2], dtype=bool), "components": 0, "max_step_m": None, "usable_pixels": int((valid & (err <= MAX_STEREO_ERROR_M)).sum())}
    if valid.sum() < 50:
        return empty
    up = -xyz[..., 2]
    n0, e0 = xyz[..., 0][valid].min(), xyz[..., 1][valid].min()
    r = np.zeros(xyz.shape[:2], dtype=int); c = np.zeros(xyz.shape[:2], dtype=int)
    r[valid] = np.floor((xyz[..., 0][valid] - n0) / GROUND_CELL_M).astype(int)
    c[valid] = np.floor((xyz[..., 1][valid] - e0) / GROUND_CELL_M).astype(int)
    grid = np.full((r[valid].max() + 1, c[valid].max() + 1), np.inf)
    np.minimum.at(grid, (r[valid], c[valid]), up[valid])
    grid[~np.isfinite(grid)] = np.nan
    k = int(round(GROUND_WINDOW_M / GROUND_CELL_M))

    def ground_of(window):
        finite = window[np.isfinite(window)]
        return np.percentile(finite, GROUND_PERCENTILE) if finite.size >= 6 else np.nan

    ground = ndimage.generic_filter(grid, ground_of, size=2 * k + 1, mode="constant", cval=np.nan)
    step = np.full(xyz.shape[:2], np.nan)
    step[valid] = up[valid] - ground[r[valid], c[valid]]
    hazard = valid & (err <= MAX_STEREO_ERROR_M) & np.isfinite(step) & (step >= step_m)
    labels, count = ndimage.label(hazard)
    usable_steps = step[valid & (err <= MAX_STEREO_ERROR_M) & np.isfinite(step)]
    max_step = float(usable_steps.max()) if usable_steps.size else None
    if count == 0:
        return {**empty, "max_step_m": max_step}
    sizes = ndimage.sum(hazard, labels, range(1, count + 1))
    keep_ids = []
    for cid in np.where(sizes >= MIN_COMPONENT_PX)[0] + 1:
        pts = xyz[labels == cid][:, :2].astype(np.float64)
        extent = float(np.max(np.percentile(pts, 97, axis=0) - np.percentile(pts, 3, axis=0)))
        area = len(np.unique(np.floor(pts / 0.1).astype(np.int64), axis=0)) * 0.01
        if extent <= MAX_FOOTPRINT_M and area <= MAX_FOOTPRINT_AREA_M2:
            keep_ids.append(cid)
    keep_ids = np.array(keep_ids, dtype=int)
    keep = np.isin(labels, keep_ids)
    return {"mask": keep, "components": int(len(keep_ids)), "max_step_m": max_step, "usable_pixels": empty["usable_pixels"]}


def hazard_overlay(png: bytes, hazard: np.ndarray) -> bytes:
    image = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(np.float32)
    if hazard.shape == image.shape[:2] and hazard.any():
        image[hazard] = image[hazard] * .25 + np.array([60, 225, 255], dtype=np.float32) * .75
    out = io.BytesIO(); Image.fromarray(image.astype(np.uint8)).save(out, format="PNG")
    return out.getvalue()


def localize_obstacles(frame: dict, mask: np.ndarray, confidence: np.ndarray, meta: dict, step_m: float = 0.15) -> dict:
    registration = frame.get("registration",{})
    if not registration.get("verified"):
        return {"accepted":False,"reason":"Independent orbital registration has not been verified; map unchanged."}
    error = registration.get("horizontal_error_bound_m")
    if error is None or not np.isfinite(error) or error < 0 or error > 2 or not registration.get("evidence"):
        return {"accepted":False,"reason":"Missing, unsupported or excessive registration error bound; map unchanged."}
    if not frame.get("xyz_path") or not frame.get("places"):
        return {"accepted":False,"reason":"Missing XYZ product or frame transform; map unchanged."}
    if not registration_evidence_matches(frame):
        return {"accepted":False,"reason":"Registration evidence is missing, mismatched or stale; map unchanged."}
    xyz = np.load(DATA_DIR / frame["xyz_path"])
    if xyz.shape != (*mask.shape,3):
        return {"accepted":False,"reason":"Reference image and XYZ geometry differ; map unchanged."}
    # Obstacles come from measured stereo geometry. The learned big_rock class has
    # insufficient held-out precision, so it never asserts occupied map cells.
    xye = np.load(DATA_DIR / frame["xye_path"]) if frame.get("xye_path") else None
    found = stereo_hazards(xyz, xye, step_m)
    classes = model_status().get("classes",[])
    model_big_rock = int(((mask==classes.index("big_rock")) & (confidence>=model_status().get("threshold",.7))).sum()) if "big_rock" in classes else 0
    stats = {"obstacle_source":"stereo_geometry","step_threshold_m":step_m,"max_stereo_error_m":MAX_STEREO_ERROR_M,
             "hazard_pixels":int(found["mask"].sum()),"hazard_components":found["components"],
             "max_step_m":None if found["max_step_m"] is None else round(found["max_step_m"],3),
             "usable_stereo_pixels":found["usable_pixels"],"model_big_rock_pixels_not_used":model_big_rock,
             "hazard_mask":found["mask"]}
    points = xyz[found["mask"]]
    if len(points)<MIN_COMPONENT_PX:
        return {"accepted":True,"points_rc":np.empty((0,2)),"registration_error_m":error,**stats,
                "reason":f"No stereo-measured step >= {step_m:.2f} m with stereo error <= {MAX_STEREO_ERROR_M:.2f} m."}
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
    return {"accepted":True,"points_rc":rc,"registration_error_m":error,**stats,
            "reason":f"Stereo-measured steps >= {step_m:.2f} m placed through the registered PLACES transform.","support_pixels":len(points)}
