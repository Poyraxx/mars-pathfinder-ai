"""Add more real, pixel-matched MSL Navcam RAD/XYZ/XYE frames from Sol 57.

Uses the same PDS decoder and corrected PLACES SITE->global transform as
prepare_pds.py. Frames are grouped into an ordered drive sequence and the
Rocknest Navcam panorama (NCAM00445/00446). Geometry is stored on a 256x256
grid (block-centre XYZ samples, never averaged across depth edges); the full
resolution radiance stretch is stored separately for display.

Registration is NOT set here; run scripts/register_image_correlation.py.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

import prepare_pds as P

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "prepared"
GRID = 256
DRIVE = ["402556045", "402556100", "402556155", "402556263", "402556366", "402556425"]
PANORAMA_SEQUENCES = ("NCAM00445", "NCAM00446")


def products(listing: str) -> list[tuple[str, str]]:
    rad = sorted(set(re.findall(r'href="(NLA_\d+RAD_[DF][^"/]+)\.LBL"', listing)))
    xyz = set(re.findall(r'href="(NLA_\d+XYZ_[DF][^"/]+)\.LBL"', listing))
    xye = set(re.findall(r'href="(NLA_\d+XYE_[DF][^"/]+)\.LBL"', listing))
    chosen = []
    for rad_id in rad:
        xyz_id = rad_id.replace("RAD_", "XYZ_"); xye_id = rad_id.replace("RAD_", "XYE_")
        if xyz_id not in xyz or xye_id not in xye:
            continue
        sclk = rad_id[4:13]
        if sclk in DRIVE:
            chosen.append(("drive", xyz_id))
        elif any(s in rad_id for s in PANORAMA_SEQUENCES):
            chosen.append(("panorama", xyz_id))
    return chosen


def block_centre(a: np.ndarray, k: int) -> np.ndarray:
    return a[k // 2::k, k // 2::k][:GRID, :GRID]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--limit", type=int, default=0); args = parser.parse_args()
    cache = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "mars-pathfinder" / "raw" / "pds"
    listing = P.fetch(P.BASE, cache / "sol57-directory.html").decode("utf-8", errors="replace")
    docs = {n: (cache / Path(n).name).read_text("utf-8-sig") for n in
            ["data_localizations/localized_pos_demv2.csv", "data_localizations/localized_interp_demv2.csv"]}
    pos = list(csv.DictReader(io.StringIO(docs["data_localizations/localized_pos_demv2.csv"])))
    interp = list(csv.DictReader(io.StringIO(docs["data_localizations/localized_interp_demv2.csv"])))
    frames_path = DATA / "frames.json"
    frames = json.loads(frames_path.read_text("utf-8"))
    existing = {f["id"] for f in frames}
    chosen = products(listing)
    if args.limit:
        chosen = chosen[:args.limit]
    order = {"drive": 0, "panorama": 0}
    for sequence, xyz_id in chosen:
        order[sequence] += 1
        if xyz_id in existing:
            for f in frames:
                if f["id"] == xyz_id:
                    f["sequence"] = sequence; f["sequence_index"] = order[sequence]
            continue
        arrays, labels, prods = {}, {}, []
        for kind in ["RAD", "XYZ", "XYE"]:
            pid = xyz_id.replace("XYZ_", kind + "_")
            lb = P.fetch(P.BASE + pid + ".LBL", cache / (pid + ".LBL")); ib = P.fetch(P.BASE + pid + ".IMG", cache / (pid + ".IMG"))
            values, layout = P.decode_pds(lb, ib)
            arrays[kind] = values; labels[kind] = lb.decode("ascii")
            prods.append({"product_id": pid, "label_url": P.BASE + pid + ".LBL", "image_url": P.BASE + pid + ".IMG",
                          "image_sha256": P.sha(ib), "label_sha256": P.sha(lb), "layout": layout})
        if arrays["RAD"].shape[1:] != arrays["XYZ"].shape[1:] or arrays["XYZ"].shape != arrays["XYE"].shape:
            raise ValueError(f"{xyz_id}: pixel correspondence mismatch")
        lab = labels["XYZ"]
        derived = P.group(lab, "DERIVED_IMAGE_PARMS")
        if P.field(derived, "REFERENCE_COORD_SYSTEM_NAME") != "SITE_FRAME":
            raise ValueError(f"{xyz_id}: XYZ not in SITE_FRAME")
        site = int(P.field(derived, "REFERENCE_COORD_SYSTEM_INDEX"))
        motion = [int(v) for v in re.search(r"ROVER_MOTION_COUNTER\s*=\s*\(([^)]+)\)", lab).group(1).split(",")]
        site_row = next(r for r in pos if r["frame"] == "SITE" and int(r["site"]) == site)
        if [float(site_row[k]) for k in ["quat_s", "quat_v1", "quat_v2", "quat_v3"]] != [1., 0., 0., 0.]:
            raise ValueError("Non-identity SITE orientation")
        rows = [r for r in interp if r["frame"] == "ROVER" and int(r["site"]) == site and int(r["drive"]) == motion[1]]
        view = "localized_interp_demv2"
        if not rows:
            rows = [r for r in pos if r["frame"] == "ROVER" and int(r["site"]) == site and int(r["drive"]) == motion[1]]
            view = "localized_pos_demv2"
        rover = next((r for r in rows if int(r["pose"]) == motion[2]), None) or next((r for r in rows if int(r["pose"]) == -1), None)
        if rover is None:
            print("skip (no PLACES pose):", xyz_id); continue
        tel = P.group(lab, "ROVER_COORD_SYSTEM_PARMS")
        if P.field(tel, "REFERENCE_COORD_SYSTEM_NAME") != "SITE_FRAME":
            raise ValueError("Unexpected telemetry parent frame")
        old_offset = P.vector(tel, "ORIGIN_OFFSET_VECTOR")
        old_rot = P.rotation_wxyz(P.vector(tel, "ORIGIN_ROTATION_QUATERNION"))
        new_rot = P.rotation_wxyz(np.array([float(rover[k]) for k in ["quat_s", "quat_v1", "quat_v2", "quat_v3"]]))
        delta = new_rot @ old_rot.T
        rover_ned = np.array([float(rover["northing"]), float(rover["easting"]), -float(rover["elevation"])])
        translation = rover_ned - delta @ old_offset
        xyz = np.moveaxis(arrays["XYZ"], 0, -1); xye = np.moveaxis(arrays["XYE"], 0, -1)
        k = xyz.shape[0] // GRID
        if k < 1 or xyz.shape[0] % GRID or xyz.shape[1] != xyz.shape[0]:
            raise ValueError(f"{xyz_id}: unsupported size {xyz.shape}")
        xyz_s = block_centre(xyz, k).astype(np.float32); xye_s = block_centre(xye, k).astype(np.float32)
        valid = np.isfinite(xyz_s).all(-1) & np.isfinite(xye_s).all(-1) & ~np.all(xyz_s == 0, -1) & (np.abs(xyz_s).max(-1) < 1e6)
        xyz_s[~valid] = 0
        rad = arrays["RAD"][0].astype(np.float64)
        finite = rad[np.isfinite(rad) & (rad > 0)]
        lo, hi = np.percentile(finite, [1, 99])
        disp = np.uint8(np.clip((rad - lo) / (hi - lo), 0, 1) * 255)
        d = DATA / "pds" / xyz_id; d.mkdir(parents=True, exist_ok=True)
        np.save(d / "xyz.npy", xyz_s, allow_pickle=False); np.save(d / "xye.npy", xye_s, allow_pickle=False); np.save(d / "valid.npy", valid, allow_pickle=False)
        full = Image.fromarray(disp).convert("RGB")
        full.resize((GRID, GRID), Image.Resampling.BOX).save(d / "reference.png")
        if k > 1:
            full.save(d / "display.png", optimize=True)
        az = re.search(r"INSTRUMENT_AZIMUTH\s*=\s*([-\d.]+)", lab); el = re.search(r"INSTRUMENT_ELEVATION\s*=\s*([-\d.]+)", lab)
        rel = lambda p: str(p.relative_to(DATA)).replace("\\", "/")
        frames.append({
            "id": xyz_id, "number": len(frames) + 1, "mission": "MSL", "instrument": "NAV_LEFT_A", "sol": 57,
            "acquired_at": P.field(lab, "START_TIME") + "Z", "site": site, "drive": motion[1], "pose": motion[2],
            "sequence": sequence, "sequence_index": order[sequence],
            "instrument_azimuth_deg": float(az.group(1)) if az else None, "instrument_elevation_deg": float(el.group(1)) if el else None,
            "native_shape": list(xyz.shape[:2]), "shape": [GRID, GRID], "grid_sampling": f"block-centre every {k} px" if k > 1 else "native",
            "reference_image_product_id": prods[0]["product_id"], "products": prods,
            "valid_pixel_fraction": float(valid.mean()),
            "display_transform": {"method": "percentile_stretch_1_99", "dn_low": float(lo), "dn_high": float(hi), "scientific_radiance": False},
            "frame": {"name": "SITE_FRAME", "index": site, "axes": ["north", "east", "down"], "units": "m"},
            "places": {"view": view, "row": rover, "site_row": site_row,
                       "telemetry_rover_origin_in_site_m": old_offset.tolist(), "corrected_rover_origin_global_ned_m": rover_ned.tolist(),
                       "archive_site_to_global_ned_rotation": delta.tolist(), "archive_site_to_global_ned_translation_m": translation.tolist(),
                       "transform": "global_NED = rotation @ archived_SITE_XYZ + translation; H=-global_NED[2]",
                       "angular_conversion": "lat_deg=N/R*180/pi; lon_deg=E/R*180/pi",
                       "correction_note": "Corrected rover origin replaces telemetry rover anchor."},
            "region_id": "gale", "image_path": rel(d / "reference.png"), "display_path": rel(d / "display.png") if k > 1 else None,
            "xyz_path": rel(d / "xyz.npy"), "xye_path": rel(d / "xye.npy"),
            "registration": {"verified": False, "horizontal_error_bound_m": None, "evidence": None,
                             "reason": "Registration not yet measured."},
            "registration_verified": False, "enabled_for_map_update": False,
            "source_url": prods[0]["label_url"]})
        print(f"added {sequence} #{order[sequence]} {xyz_id} valid={valid.mean():.2f}", flush=True)
    frames_path.write_text(json.dumps(frames, indent=2), "utf-8")


if __name__ == "__main__":
    main()
