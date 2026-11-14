"""Measure camera/orbital registration of archived Navcam frames by image correlation.

Method (fixed before evaluation; do not tune per frame):

1. Every valid camera pixel is placed on the 25 cm PLACES HiRISE ortho grid using its
   archived stereo XYZ (SITE frame) and the documented corrected SITE->global NED affine.
   This placement is only the starting hypothesis.
2. Camera brightness (per-frame z-score) and HiRISE DN are band-pass filtered
   (difference of Gaussians) to suppress illumination/photometric differences.
3. The zero-mean normalized cross-correlation (NCC) is evaluated for every horizontal
   shift within +-8 m. The global best shift is the measured horizontal residual of the
   archived transform. Nothing is fitted to the archived transform itself.
4. Significance: statistic T = max NCC within +-1 m of a placement. T at the archived
   placement is compared with T at 2000 random placements of the same patch at least 15 m
   away (seeded), giving an empirical p-value.
5. Robustness: steps 2-4 are repeated for four fixed band-pass settings.

Acceptance (all settings): p <= 0.01 and global best shift <= 1.5 m.
Error bound reported to the planner: max |best shift| over settings + 0.5 m (two ortho
pixels for quantization/HiRISE resolution). This is an empirical research tolerance, not a
probabilistic confidence interval. The three frames overlap; they are not independent.

Run with --write to store evidence and update frames.json; otherwise it is a dry run.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "prepared"
SETTINGS = [(0.75, 4.0), (1.0, 6.0), (1.5, 8.0), (1.0, 10.0)]
SEARCH_M = 8.0
LOCAL_M = 1.0
NULL_N = 2000
NULL_MIN_DISTANCE_M = 15.0
SEED = 57
P_MAX = 0.01
SHIFT_MAX_M = 1.5
QUANTIZATION_MARGIN_M = 0.5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dog(a, lo, hi):
    return ndimage.gaussian_filter(a, lo) - ndimage.gaussian_filter(a, hi)


def camera_patch(frame, res, e0, n0):
    xyz = np.load(DATA / frame["xyz_path"]).astype(float)
    img = np.asarray(Image.open(DATA / frame["image_path"]).convert("L"), dtype=float)
    valid = np.isfinite(xyz).all(-1) & (xyz != 0).any(-1)
    p = frame["places"]
    rot = np.array(p["archive_site_to_global_ned_rotation"]); tr = np.array(p["archive_site_to_global_ned_translation_m"])
    ned = xyz[valid] @ rot.T + tr
    b = img[valid]; b = (b - b.mean()) / (b.std() + 1e-9)
    c = np.floor((ned[:, 1] - e0) / res).astype(int); r = np.floor((n0 - ned[:, 0]) / res).astype(int)
    r0, c0 = int(r.min()) - 8, int(c.min()) - 8
    h, w = int(r.max()) - r0 + 9, int(c.max()) - c0 + 9
    s = np.zeros((h, w)); n = np.zeros((h, w))
    np.add.at(s, (r - r0, c - c0), b); np.add.at(n, (r - r0, c - c0), 1)
    m0 = n > 0
    cam = np.where(m0, s / np.maximum(n, 1), 0.0)
    wts = ndimage.gaussian_filter(m0.astype(float), 1.0)
    camf = ndimage.gaussian_filter(cam, 1.0) / np.maximum(wts, 1e-6)
    mask = ndimage.binary_erosion(m0, iterations=2) & (n >= 2)
    return camf, mask, r0, c0


def ncc_at(pa, na, ob, mask, y, x):
    ph, pw = mask.shape
    if y < 0 or x < 0 or y + ph > ob.shape[0] or x + pw > ob.shape[1]:
        return None
    b = ob[y:y + ph, x:x + pw][mask]; b = b - b.mean()
    return float((pa * b).sum() / (na * np.sqrt((b * b).sum()) + 1e-12))


def local_max(pa, na, ob, mask, y, x, radius):
    best = -1.0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            v = ncc_at(pa, na, ob, mask, y + dy, x + dx)
            if v is None:
                return None
            best = max(best, v)
    return best


def evaluate(frame, ortho, res, e0, n0):
    camf, mask, r0, c0 = camera_patch(frame, res, e0, n0)
    search = int(round(SEARCH_M / res)); local = int(round(LOCAL_M / res))
    out = []
    for lo, hi in SETTINGS:
        ob = dog(ortho, lo, hi)
        pa = dog(camf, lo, hi)[mask]; pa = pa - pa.mean(); na = float(np.sqrt((pa * pa).sum()))
        surf = np.full((2 * search + 1, 2 * search + 1), np.nan)
        for dy in range(-search, search + 1):
            for dx in range(-search, search + 1):
                v = ncc_at(pa, na, ob, mask, r0 + dy, c0 + dx)
                surf[dy + search, dx + search] = np.nan if v is None else v
        iy, ix = np.unravel_index(np.nanargmax(surf), surf.shape)
        shift_e, shift_s = (ix - search) * res, (iy - search) * res
        observed = local_max(pa, na, ob, mask, r0, c0, local)
        rng = np.random.default_rng(SEED)
        null = []
        while len(null) < NULL_N:
            y = int(rng.integers(0, ortho.shape[0] - mask.shape[0])); x = int(rng.integers(0, ortho.shape[1] - mask.shape[1]))
            if np.hypot(y - r0, x - c0) * res < NULL_MIN_DISTANCE_M:
                continue
            v = local_max(pa, na, ob, mask, y, x, local)
            if v is not None:
                null.append(v)
        null = np.array(null)
        p = float((1 + (null >= observed).sum()) / (1 + len(null)))
        out.append({"bandpass_sigma_px": [lo, hi], "best_shift_east_m": float(shift_e), "best_shift_south_m": float(shift_s),
                    "best_shift_m": float(np.hypot(shift_e, shift_s)), "best_ncc": float(surf[iy, ix]),
                    "ncc_at_archived_placement": float(surf[search, search]), "statistic_T_within_1m": float(observed),
                    "null_n": int(len(null)), "null_p95": float(np.percentile(null, 95)), "null_p99": float(np.percentile(null, 99)),
                    "null_max": float(null.max()), "empirical_p": p})
    return out, int(mask.sum())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--write", action="store_true"); args = parser.parse_args()
    frames_path = DATA / "frames.json"
    frames = json.loads(frames_path.read_text("utf-8"))
    ortho_meta = json.loads((DATA / "pds/region-ortho.json").read_text("utf-8"))
    ortho_path = DATA / "pds" / ortho_meta["image"]
    ortho = np.asarray(Image.open(ortho_path).convert("L"), dtype=np.float64)
    res = ortho_meta["pixel_resolution_m"]; e0 = ortho_meta["upper_left_easting_m"]; n0 = ortho_meta["upper_left_northing_m"]
    summary = []
    for frame in frames:
        trials, mask_px = evaluate(frame, ortho, res, e0, n0)
        ok = all(t["empirical_p"] <= P_MAX and t["best_shift_m"] <= SHIFT_MAX_M for t in trials)
        bound = max(t["best_shift_m"] for t in trials) + QUANTIZATION_MARGIN_M
        evidence = {
            "frame_id": frame["id"], "camera_product_id": frame["reference_image_product_id"],
            "method": "image_correlation_shift_search",
            "method_description": __doc__.strip(),
            "acceptance": {"empirical_p_max": P_MAX, "best_shift_max_m": SHIFT_MAX_M, "all_settings_required": True},
            "verified": bool(ok),
            "horizontal_error_bound_m": float(bound) if ok else None,
            "error_bound_definition": "max |global best shift| over band-pass settings + 0.5 m quantization margin; empirical research tolerance, not a confidence interval",
            "camera_mask_pixels_25cm": mask_px,
            "trials": trials,
            "inputs_sha256": {"reference_png": sha256(DATA / frame["image_path"]), "xyz_npy": sha256(DATA / frame["xyz_path"]),
                              "region_ortho_png": sha256(ortho_path)},
            "orbital_source": {k: ortho_meta[k] for k in ("source_lid", "version", "image_url", "range_sha256", "pixel_resolution_m",
                                                          "upper_left_easting_m", "upper_left_northing_m")},
            "limitations": [
                "Frames are 256x256 TRAV Navcam images covering about 3-11 m range; the three frames overlap and are not independent.",
                "PLACES localized rover positions were themselves derived against the HiRISE base map; this test checks the full XYZ + corrected affine + projection chain against image content, not the base-map geodesy.",
                "Different illumination/epochs between Navcam and HiRISE; band-pass filtering reduces but does not remove this.",
                "Vertical agreement is not tested here; stereo vs orbital DEM differs by about 1.1 m.",
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        summary.append({"frame": frame["id"], "verified": evidence["verified"], "bound_m": evidence["horizontal_error_bound_m"],
                        "shifts_m": [round(t["best_shift_m"], 2) for t in trials], "p": [t["empirical_p"] for t in trials]})
        if args.write:
            path = DATA / "pds" / frame["id"] / "registration-evidence.json"
            path.write_text(json.dumps(evidence, indent=2), "utf-8")
            rel = str(path.relative_to(DATA)).replace("\\", "/")
            if ok:
                frame["registration"] = {"verified": True, "horizontal_error_bound_m": evidence["horizontal_error_bound_m"],
                                         "method": evidence["method"], "evidence": rel, "evidence_sha256": sha256(path),
                                         "reason": "Camera content matches HiRISE at the archived placement (image-correlation shift search, empirical null test)."}
                frame["registration_verified"] = True
                frame["enabled_for_map_update"] = True
                frame["registration_status"] = (f"Image-correlation registration: residual <= {bound - QUANTIZATION_MARGIN_M:.2f} m, "
                                                 f"planner bound {bound:.2f} m; empirical tolerance, not a confidence interval.")
            else:
                frame["registration"] = {"verified": False, "horizontal_error_bound_m": None, "method": evidence["method"],
                                         "evidence": rel, "reason": "Image-correlation registration did not meet the fixed acceptance criteria."}
                frame["registration_verified"] = False
                frame["enabled_for_map_update"] = False
    if args.write:
        frames_path.write_text(json.dumps(frames, indent=2), "utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
