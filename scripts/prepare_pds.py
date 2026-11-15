"""Prepare real, pixel-matched MSL Navcam/XYZ archive candidates.

This script establishes archive identity and a documented PLACES frame chain.
It deliberately does NOT certify independent camera/orbital registration.
Run with Python, NumPy and Pillow. No models or simulated geometry are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://planetarydata.jpl.nasa.gov/img/data/msl/MSLNAV_1XXX/DATA_V1/SOL00057/"
PLACES = "https://planetarydata.jpl.nasa.gov/w10n/msl/msl_places/"
FIRST_XYZ = "NLA_402556045XYZ_D0043480TRAV00022M1"


def fetch(url: str, destination: Path) -> bytes:
    if destination.exists():
        return destination.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent": "MarsPathfinderResearch/1.0"}), timeout=90) as response:
                payload = response.read()
            destination.write_bytes(payload)
            return payload
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def field(text: str, name: str, default: str | None = None) -> str:
    match = re.search(r"(?m)^\s*" + re.escape(name) + r"\s*=\s*([^\r\n]+)", text)
    if not match:
        if default is not None:
            return default
        raise ValueError(f"Missing PDS field {name}")
    return match.group(1).strip().strip('"')


def decode_pds(label_bytes: bytes, image_bytes: bytes) -> tuple[np.ndarray, dict]:
    """Read this archive's fixed-record detached PDS3/BSQ image accurately."""
    label = label_bytes.decode("ascii")
    match = re.search(r"(?ms)^OBJECT\s*=\s*IMAGE\s*$\s*(.*?)^END_OBJECT\s*=\s*IMAGE\s*$", label)
    if not match:
        raise ValueError("Missing IMAGE object")
    image = match.group(1)
    pointer = re.search(r'\^IMAGE\s*=\s*\(\s*"([^"]+)"\s*,\s*(\d+)\s*(<BYTES>)?\s*\)', label)
    if not pointer:
        raise ValueError("Unsupported IMAGE pointer; refusing to guess offset")
    filename, position, byte_units = pointer.groups()
    record_bytes = int(field(label, "RECORD_BYTES"))
    offset = (int(position) - 1) * (1 if byte_units else record_bytes)
    height = int(field(image, "LINES"))
    width = int(field(image, "LINE_SAMPLES"))
    bands = int(field(image, "BANDS", "1"))
    bits = int(field(image, "SAMPLE_BITS"))
    sample_type = field(image, "SAMPLE_TYPE")
    storage = field(image, "BAND_STORAGE_TYPE", "BAND_SEQUENTIAL")
    if storage != "BAND_SEQUENTIAL":
        raise ValueError(f"Unsupported storage: {storage}")
    if int(field(image, "LINE_PREFIX_BYTES", "0")) or int(field(image, "LINE_SUFFIX_BYTES", "0")):
        raise ValueError("Line padding must be handled explicitly")
    endian = "<" if sample_type.startswith(("LSB", "PC")) else ">"
    kind = "f" if "REAL" in sample_type else ("u" if "UNSIGNED" in sample_type else "i")
    dtype = np.dtype(endian + kind + str(bits // 8))
    count = height * width * bands
    if offset + count * dtype.itemsize > len(image_bytes):
        raise ValueError("Truncated image payload")
    values = np.frombuffer(image_bytes, dtype=dtype, offset=offset, count=count).reshape(bands, height, width)
    values = values.astype(np.float32)
    scale = float(field(image, "SCALING_FACTOR", "1").split()[0])
    add = float(field(image, "OFFSET", "0").split()[0])
    values = values * scale + add
    return values, {
        "filename": filename, "shape_bands_lines_samples": [bands, height, width],
        "dtype_on_disk": dtype.str, "offset_bytes": offset, "record_bytes": record_bytes,
        "scaling_factor": scale, "offset": add, "band_storage": storage,
    }


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def group(text: str, name: str) -> str:
    match = re.search(r"(?ms)^GROUP\s*=\s*" + name + r"\s*$\s*(.*?)^END_GROUP\s*=\s*" + name + r"\s*$", text)
    if not match:
        raise ValueError(f"Missing PDS group {name}")
    return match.group(1)


def vector(text: str, name: str) -> np.ndarray:
    match = re.search(re.escape(name) + r"\s*=\s*\(([^)]+)\)", text)
    if not match:
        raise ValueError(f"Missing vector {name}")
    return np.array([float(value) for value in match.group(1).split(",")], dtype=np.float64)


def rotation_wxyz(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def prepare_orbital_crop(cache: Path, output: Path, candidates: list[dict]) -> dict:
    """Read a small true HiRISE cutout from the PLACES base map via HTTP Range.

    The archive includes the very base map the localization documentation names.
    Reading a crop avoids downloading its entire approximately1.5GB image.
    """
    xml_url = PLACES + "data_maps/msl_orbital_map.xml"
    xml_bytes = fetch(xml_url, cache / "msl_orbital_map.xml")
    root = ET.fromstring(xml_bytes)
    ns = {"p": "http://pds.nasa.gov/pds4/pds/v1", "c": "http://pds.nasa.gov/pds4/cart/v1"}
    array = root.find(".//p:Array_2D_Image", ns)
    dtype = array.findtext("p:Element_Array/p:data_type", namespaces=ns)
    if dtype != "UnsignedByte" or array.findtext("p:axis_index_order", namespaces=ns) != "Last Index Fastest":
        raise ValueError("Unsupported orbital map layout")
    axes = {node.findtext("p:axis_name", namespaces=ns): int(node.findtext("p:elements", namespaces=ns)) for node in array.findall("p:Axis_Array", ns)}
    height, width = axes["Line"], axes["Sample"]
    offset = int(array.findtext("p:offset", namespaces=ns))
    transform = root.find(".//c:Geo_Transformation", ns)
    left = float(transform.findtext("c:upperleft_corner_x", namespaces=ns))
    top = float(transform.findtext("c:upperleft_corner_y", namespaces=ns))
    resolution = float(root.findtext(".//c:pixel_resolution_x", namespaces=ns))
    candidate = candidates[0]
    xyz = np.load(output / candidate["xyz_file"])
    valid = np.load(output / candidate["valid_file"])
    matrix = np.array(candidate["places"]["archive_site_to_global_ned_rotation"])
    translation = np.array(candidate["places"]["archive_site_to_global_ned_translation_m"])
    projected = xyz[valid] @ matrix.T + translation
    north, east = np.median(projected[:, :2], axis=0)
    size = 256
    row0 = max(0, min(height-size, int((top-north)/resolution)-size//2))
    col0 = max(0, min(width-size, int((east-left)/resolution)-size//2))
    start = offset + row0 * width
    end = offset + (row0 + size) * width - 1
    url = "https://planetarydata.jpl.nasa.gov/img/data/msl/msl_places/data_maps/msl_orbital_map.img"
    cache_path = cache / f"orbital_map_rows_{row0}_{row0+size-1}.bin"
    if cache_path.exists():
        payload = cache_path.read_bytes()
    else:
        request = Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "MarsPathfinderResearch/1.0"})
        with urlopen(request, timeout=90) as response:
            if response.status != 206:
                raise ValueError("Server ignored range; refusing whole orbital image download")
            payload = response.read(end-start+2)
        if len(payload) != end-start+1:
            raise ValueError("Orbital row range length mismatch")
        cache_path.write_bytes(payload)
    raster = np.frombuffer(payload, dtype=np.uint8).reshape(size, width)[:, col0:col0+size]
    Image.fromarray(raster).save(output / "orbital-crop.png")
    metadata = {"source_lid": root.findtext(".//p:logical_identifier", namespaces=ns), "version": root.findtext(".//p:version_id", namespaces=ns), "image_url": url, "label_url": xml_url, "label_sha256": sha(xml_bytes), "range_sha256": sha(payload), "source_byte_range": [start, end], "source_image_shape": [height, width], "source_pixel_window": {"row": row0, "column": col0, "height": size, "width": size}, "pixel_resolution_m": resolution, "upper_left_easting_m": left+col0*resolution, "upper_left_northing_m": top-row0*resolution, "image": "orbital-crop.png", "purpose": "Actual HiRISE PLACES base-map context for independent registration checks; residuals not yet measured", "registration_verified": False}
    (output / "orbital-crop.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared true PLACES HiRISE crop: {size}x{size} at {resolution}m/pixel", flush=True)
    return metadata


def prepare_dem_crop(cache: Path, output: Path, candidates: list[dict]) -> dict:
    """Acquire a native1m crop of the PLACES included Gale DEM, preserving holes."""
    xml_url = PLACES + "data_maps/msl_orbital_dem.xml"
    xml_bytes = fetch(xml_url, cache / "msl_orbital_dem.xml")
    root = ET.fromstring(xml_bytes)
    ns = {"p": "http://pds.nasa.gov/pds4/pds/v1", "c": "http://pds.nasa.gov/pds4/cart/v1"}
    array = root.find(".//p:Array_2D_Image", ns)
    dtype_name = array.findtext("p:Element_Array/p:data_type", namespaces=ns)
    if dtype_name != "IEEE754MSBSingle" or array.findtext("p:axis_index_order", namespaces=ns) != "Last Index Fastest":
        raise ValueError("Unsupported PLACES DEM layout")
    axes = {node.findtext("p:axis_name", namespaces=ns): int(node.findtext("p:elements", namespaces=ns)) for node in array.findall("p:Axis_Array", ns)}
    height, width = axes["Line"], axes["Sample"]
    offset = int(array.findtext("p:offset", namespaces=ns))
    transform = root.find(".//c:Geo_Transformation", ns)
    left = float(transform.findtext("c:upperleft_corner_x", namespaces=ns))
    top = float(transform.findtext("c:upperleft_corner_y", namespaces=ns))
    resolution = float(root.findtext(".//c:pixel_resolution_x", namespaces=ns))
    anchor = candidates[0]["places"]["corrected_rover_origin_global_ned_m"]
    size = 512
    row0 = max(0, min(height-size, int((top-anchor[0])/resolution)-size//2))
    col0 = max(0, min(width-size, int((anchor[1]-left)/resolution)-size//2))
    start = offset + row0 * width * 4
    end = offset + (row0 + size) * width * 4 - 1
    url = "https://planetarydata.jpl.nasa.gov/img/data/msl/msl_places/data_maps/msl_orbital_dem.img"
    cache_path = cache / f"orbital_dem_rows_{row0}_{row0+size-1}.bin"
    if cache_path.exists():
        payload = cache_path.read_bytes()
    else:
        request = Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "MarsPathfinderResearch/1.0"})
        with urlopen(request, timeout=90) as response:
            if response.status != 206:
                raise ValueError("Server ignored DEM range; refusing whole image download")
            payload = response.read(end-start+2)
        if len(payload) != end-start+1:
            raise ValueError("DEM row range length mismatch")
        cache_path.write_bytes(payload)
    values = np.frombuffer(payload, dtype=">f4").reshape(size, width)[:, col0:col0+size].astype(np.float32)
    special = {node.tag.rsplit("}", 1)[-1]: node.text for node in array.findall("p:Special_Constants/*", ns)}
    valid = np.isfinite(values) & (np.abs(values) < 1e6)
    for constant in special.values():
        try:
            valid &= values != float(constant)
        except (TypeError, ValueError):
            pass
    values[~valid] = np.nan
    np.save(output / "dem-crop.npy", values, allow_pickle=False)
    np.save(output / "dem-crop-valid.npy", valid, allow_pickle=False)
    finite = values[valid]
    metadata = {"source_lid": root.findtext(".//p:logical_identifier", namespaces=ns), "version": root.findtext(".//p:version_id", namespaces=ns), "image_url": url, "label_url": xml_url, "label_sha256": sha(xml_bytes), "range_sha256": sha(payload), "source_byte_range": [start, end], "source_image_shape": [height, width], "source_pixel_window": {"row": row0, "column": col0, "height": size, "width": size}, "pixel_resolution_m": resolution, "upper_left_easting_m": left+col0*resolution, "upper_left_northing_m": top-row0*resolution, "array": "dem-crop.npy", "valid_mask": "dem-crop-valid.npy", "height_reference": "Gale PLACES orbital DEM areoid elevation, meters", "axis_order": ["southward_row", "eastward_column"], "projection": "Equirectangular", "sphere_radius_m": 3396190.0, "center_longitude_deg": 0.0, "standard_parallel_deg": 0.0, "latitude_type": "planetocentric", "special_constants": special, "nodata_policy": "Preserve declared special constants, nonfinite values and abs(value)>=1e6 as NaN; no hole filling", "valid_fraction": float(valid.mean()), "minimum_elevation_m": float(finite.min()) if finite.size else None, "maximum_elevation_m": float(finite.max()) if finite.size else None, "zero_count": int(np.count_nonzero(values == 0)), "registration_verified": False, "note": "Native1m included PLACES DEM; metadata alignment only, independent control residuals still unmeasured"}
    (output / "dem-crop.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared PLACES DEM: {size}x{size} at {resolution}m/pixel, valid={valid.mean():.3f}", flush=True)
    return metadata


def prepare_region_ortho(cache: Path, output: Path, dem: dict) -> dict:
    """Acquire native25cm regional imagery matching the DEM to pixel precision."""
    xml_url = PLACES + "data_maps/msl_orbital_map.xml"
    xml_bytes = fetch(xml_url, cache / "msl_orbital_map.xml")
    root = ET.fromstring(xml_bytes)
    ns = {"p": "http://pds.nasa.gov/pds4/pds/v1", "c": "http://pds.nasa.gov/pds4/cart/v1"}
    array = root.find(".//p:Array_2D_Image", ns)
    if array.findtext("p:Element_Array/p:data_type", namespaces=ns) != "UnsignedByte":
        raise ValueError("Unsupported regional ortho data type")
    if array.findtext("p:axis_index_order", namespaces=ns) != "Last Index Fastest":
        raise ValueError("Unsupported regional ortho axis layout")
    axes = {node.findtext("p:axis_name", namespaces=ns): int(node.findtext("p:elements", namespaces=ns)) for node in array.findall("p:Axis_Array", ns)}
    height, width = axes["Line"], axes["Sample"]
    offset = int(array.findtext("p:offset", namespaces=ns))
    transform = root.find(".//c:Geo_Transformation", ns)
    left = float(transform.findtext("c:upperleft_corner_x", namespaces=ns))
    top = float(transform.findtext("c:upperleft_corner_y", namespaces=ns))
    resolution = float(root.findtext(".//c:pixel_resolution_x", namespaces=ns))
    target_left = dem["upper_left_easting_m"]
    target_top = dem["upper_left_northing_m"]
    extent = dem["source_pixel_window"]["width"] * dem["pixel_resolution_m"]
    size = int(round(extent / resolution))
    row0 = int(round((top-target_top) / resolution))
    col0 = int(round((target_left-left) / resolution))
    if not (0 <= row0 <= height-size and 0 <= col0 <= width-size):
        raise ValueError("Requested regional image window outside source footprint")
    start = offset + row0 * width
    end = offset + (row0+size) * width - 1
    url = "https://planetarydata.jpl.nasa.gov/img/data/msl/msl_places/data_maps/msl_orbital_map.img"
    cache_path = cache / f"orbital_region_rows_{row0}_{row0+size-1}.bin"
    if cache_path.exists():
        payload = cache_path.read_bytes()
    else:
        request = Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "MarsPathfinderResearch/1.0"})
        with urlopen(request, timeout=90) as response:
            if response.status != 206:
                raise ValueError("Server ignored regional image range; refusing whole download")
            payload = response.read(end-start+2)
        if len(payload) != end-start+1:
            raise ValueError("Regional ortho byte range length mismatch")
        cache_path.write_bytes(payload)
    raster = np.frombuffer(payload, dtype=np.uint8).reshape(size, width)[:, col0:col0+size]
    Image.fromarray(raster).save(output / "region-ortho.png")
    actual_left = left+col0*resolution
    actual_top = top-row0*resolution
    actual_right = actual_left+size*resolution
    actual_bottom = actual_top-size*resolution
    radius = 3396190.0
    metadata = {"source_lid": root.findtext(".//p:logical_identifier", namespaces=ns), "version": root.findtext(".//p:version_id", namespaces=ns), "attribution": "NASA/JPL-Caltech; MRO HiRISE and MSL PLACES teams", "image_url": url, "label_url": xml_url, "label_sha256": sha(xml_bytes), "range_sha256": sha(payload), "png_sha256": sha((output / "region-ortho.png").read_bytes()), "source_byte_range": [start,end], "source_image_shape": [height,width], "source_pixel_window": {"row": row0,"column": col0,"height": size,"width": size}, "pixel_resolution_m": resolution, "upper_left_easting_m": actual_left, "upper_left_northing_m": actual_top, "lower_right_easting_m": actual_right, "lower_right_northing_m": actual_bottom, "requested_dem_upper_left_m": {"easting": target_left,"northing": target_top}, "native_pixel_rounding_offset_m": {"easting": actual_left-target_left,"northing": actual_top-target_top}, "render_rectangle_degrees_west_south_east_north": [math.degrees(actual_left/radius),math.degrees(actual_bottom/radius),math.degrees(actual_right/radius),math.degrees(actual_top/radius)], "image": "region-ortho.png", "projection": "Equirectangular", "sphere_radius_m": radius, "latitude_type": "planetocentric", "center_longitude_deg": 0.0, "standard_parallel_deg": 0.0, "zero_pixel_fraction": float(np.mean(raster==0)), "display_processing": "None; source8bit values preserved", "registration_verified": False, "note": "True PLACES included25cm HiRISE imagery. Bounds rounded to nearest native image pixel; independent camera/orbital registration residuals remain unmeasured."}
    (output / "region-ortho.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared regional HiRISE: {size}x{size}, {resolution}m/pixel, {len(payload)}range bytes", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--ortho-crop", action="store_true", help="Acquire true25cm HiRISE PLACES map crop with HTTP Range")
    parser.add_argument("--dem-crop", action="store_true", help="Acquire native1m PLACES DEM crop with HTTP Range")
    parser.add_argument("--region-ortho", action="store_true", help="Acquire native25cm HiRISE region matching DEM bounds")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "prepared" / "pds")
    args = parser.parse_args()
    cache = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "mars-pathfinder" / "raw" / "pds"
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    sources = []
    documents = {}
    for name in ["data_localizations/localized_pos_demv2.csv", "data_localizations/localized_interp_demv2.csv", "data_localizations/view_metadata.csv", "data_orbital/orbital_1_metadata.csv", "data_orbital/orbital_origin_global.csv"]:
        payload = fetch(PLACES + name, cache / Path(name).name)
        documents[name] = payload.decode("utf-8-sig")
        sources.append({"url": PLACES + name, "sha256": sha(payload), "bytes": len(payload)})
    localizations = list(csv.DictReader(io.StringIO(documents["data_localizations/localized_pos_demv2.csv"])))
    interpolated = list(csv.DictReader(io.StringIO(documents["data_localizations/localized_interp_demv2.csv"])))
    orbital = {row["key"]: row["value"] for row in csv.DictReader(io.StringIO(documents["data_orbital/orbital_1_metadata.csv"]))}
    radius = float(orbital["ellipsoid_radius"])
    xyz_ids = [FIRST_XYZ]
    if args.frames > 1:
        listing = fetch(BASE, cache / "sol57-directory.html").decode("utf-8", errors="replace")
        discovered = sorted(set(re.findall(r'href="(NLA_\d+XYZ_D[^"/]+)\.LBL"', listing)))
        xyz_ids = ([FIRST_XYZ] + [name for name in discovered if name != FIRST_XYZ])[:args.frames]
        if len(xyz_ids) < args.frames:
            raise ValueError("Requested matched frames not available in archive listing")
    candidates = []
    for number, xyz_id in enumerate(xyz_ids, 1):
        frame_dir = output / xyz_id
        frame_dir.mkdir(exist_ok=True)
        arrays = {}
        products = []
        labels = {}
        for product_type in ["RAD", "XYZ", "XYE"]:
            product_id = xyz_id.replace("XYZ_D", product_type + "_D")
            label_bytes = fetch(BASE + product_id + ".LBL", cache / (product_id + ".LBL"))
            image_bytes = fetch(BASE + product_id + ".IMG", cache / (product_id + ".IMG"))
            values, layout = decode_pds(label_bytes, image_bytes)
            arrays[product_type] = values
            labels[product_type] = label_bytes.decode("ascii")
            products.append({"product_id": product_id, "label_url": BASE + product_id + ".LBL", "image_url": BASE + product_id + ".IMG", "image_sha256": sha(image_bytes), "label_sha256": sha(label_bytes), "layout": layout})
        if arrays["RAD"].shape[1:] != arrays["XYZ"].shape[1:] or arrays["XYZ"].shape != arrays["XYE"].shape:
            raise ValueError("Pixel correspondence dimensions mismatch")
        xyz_label = labels["XYZ"]
        derived = group(xyz_label, "DERIVED_IMAGE_PARMS")
        if field(derived, "REFERENCE_COORD_SYSTEM_NAME") != "SITE_FRAME":
            raise ValueError("Only explicitly declared SITE_FRAME XYZ products supported")
        site = int(field(derived, "REFERENCE_COORD_SYSTEM_INDEX"))
        motion = re.search(r"ROVER_MOTION_COUNTER\s*=\s*\(([^)]+)\)", xyz_label).group(1)
        motion_indices = [int(part) for part in motion.split(",")]
        row = next(item for item in localizations if item["frame"] == "SITE" and int(item["site"]) == site)
        if [float(row[key]) for key in ["quat_s", "quat_v1", "quat_v2", "quat_v3"]] != [1., 0., 0., 0.]:
            raise ValueError("Nonidentity SITE orientation requires full frame rotation")
        pose_candidates = [item for item in interpolated if item["frame"] == "ROVER" and int(item["site"]) == site and int(item["drive"]) == motion_indices[1]]
        if not pose_candidates:
            raise ValueError("No corrected PLACES rover localization for this motion counter")
        rover_row = next((item for item in pose_candidates if int(item["pose"]) == motion_indices[2]), None)
        if rover_row is None:
            rover_row = next((item for item in pose_candidates if int(item["pose"]) == -1), None)
        if rover_row is None:
            raise ValueError("No exact or documented pose-independent PLACES row")
        telemetry_rover = group(xyz_label, "ROVER_COORD_SYSTEM_PARMS")
        if field(telemetry_rover, "REFERENCE_COORD_SYSTEM_NAME") != "SITE_FRAME":
            raise ValueError("Unexpected telemetry rover parent frame")
        old_offset = vector(telemetry_rover, "ORIGIN_OFFSET_VECTOR")
        old_rotation = rotation_wxyz(vector(telemetry_rover, "ORIGIN_ROTATION_QUATERNION"))
        corrected_rotation = rotation_wxyz(np.array([float(rover_row[key]) for key in ["quat_s", "quat_v1", "quat_v2", "quat_v3"]]))
        rotation_delta = corrected_rotation @ old_rotation.T
        corrected_rover_ned = np.array([float(rover_row["northing"]), float(rover_row["easting"]), -float(rover_row["elevation"])])
        affine_translation_ned = corrected_rover_ned - rotation_delta @ old_offset
        xyz = np.moveaxis(arrays["XYZ"], 0, -1)
        xye = np.moveaxis(arrays["XYE"], 0, -1)
        valid = np.isfinite(xyz).all(axis=-1) & np.isfinite(xye).all(axis=-1) & ~np.all(xyz == 0, axis=-1) & (np.abs(xyz).max(axis=-1) < 1e6)
        np.save(frame_dir / "xyz.npy", xyz, allow_pickle=False)
        np.save(frame_dir / "xye.npy", xye, allow_pickle=False)
        np.save(frame_dir / "valid.npy", valid, allow_pickle=False)
        radiance_dn = arrays["RAD"][0]
        finite = radiance_dn[np.isfinite(radiance_dn) & (radiance_dn > 0)]
        low, high = np.percentile(finite, [1, 99])
        if high <= low:
            raise ValueError("No display dynamic range")
        display = np.clip((radiance_dn - low) / (high - low), 0, 1)
        Image.fromarray(np.uint8(display * 255)).convert("RGB").save(frame_dir / "reference.png")
        northing, easting, elevation = [float(row[key]) for key in ["northing", "easting", "elevation"]]
        candidates.append({
            "id": xyz_id, "number": number, "mission": "MSL", "instrument": "NAV_LEFT_A", "sol": 57,
            "acquired_at": field(xyz_label, "START_TIME") + "Z", "site": site, "drive": motion_indices[1], "pose": motion_indices[2],
            "reference_image_product_id": products[0]["product_id"], "reference_image": str((frame_dir / "reference.png").relative_to(output)).replace("\\", "/"),
            "xyz_file": str((frame_dir / "xyz.npy").relative_to(output)).replace("\\", "/"), "xye_file": str((frame_dir / "xye.npy").relative_to(output)).replace("\\", "/"), "valid_file": str((frame_dir / "valid.npy").relative_to(output)).replace("\\", "/"),
            "shape": list(valid.shape), "valid_pixel_fraction": float(valid.mean()),
            "display_transform": {"method": "percentile_stretch_1_99", "dn_low": float(low), "dn_high": float(high), "scientific_radiance": False},
            "products": products,
            "frame": {"name": "SITE_FRAME", "index": site, "axes": ["north", "east", "down"], "units": "m"},
            "places": {"view": "localized_interp_demv2", "row": rover_row, "site_row": row,
                       "site_origin_projected_m": {"northing": northing, "easting": easting, "areoid_elevation": elevation},
                       "site_orientation_wxyz": [1, 0, 0, 0],
                       "telemetry_rover_origin_in_site_m": old_offset.tolist(),
                       "corrected_rover_origin_global_ned_m": corrected_rover_ned.tolist(),
                       "archive_site_to_global_ned_rotation": rotation_delta.tolist(),
                       "archive_site_to_global_ned_translation_m": affine_translation_ned.tolist(),
                       "transform": "global_NED = rotation @ archived_SITE_XYZ + translation; H=-global_NED[2]",
                       "angular_conversion": "lat_deg=N/R*180/pi; lon_deg=E/R*180/pi",
                       "site_origin_latitude_deg": math.degrees(northing / radius), "site_origin_longitude_deg": math.degrees(easting / radius),
                       "correction_note": "Corrected rover origin replaces telemetry rover anchor, removing accumulated translation drift in archive Site XYZ. Do not add the localized Site origin directly.",
                       "accuracy_note": "PLACES SIS section4.4.1.2 gives expected0.5m localized_pos accuracy relative to included base map. This is not a verified bound for interpolated in-drive frames or an independently registered HiRISE DTM."},
            "coordinate_chain_available": True, "registration_verified": False, "enabled_for_map_update": False,
            "registration_status": "Independent image/orbital control-point residuals have not been measured. No horizontal error bound is claimed.",
            "xye_note": "MSL_CAMERA_SIS section5.2.1.4: three axis-aligned estimated range-error components in meters forming an error ellipsoid; not a localization bound, covariance, confidence interval or total error budget.",
        })
        print(f"Prepared {number}: {xyz_id}, valid XYZ fraction {valid.mean():.3f}", flush=True)
    manifest = {"schema_version": "1.0", "body": "Mars", "source": "NASA PDS MSL NAVCAM RDR and MSL PLACES", "attribution": "NASA/JPL-Caltech; MSL NAVCAM and PLACES teams", "orbital_projection": orbital, "document_sources": sources, "registration_verified": False, "frames": candidates}
    if args.ortho_crop:
        manifest["orbital_crop"] = prepare_orbital_crop(cache, output, candidates)
    if args.dem_crop:
        manifest["dem_crop"] = prepare_dem_crop(cache, output, candidates)
    if args.region_ortho:
        dem_metadata = manifest.get("dem_crop")
        if dem_metadata is None:
            dem_metadata = json.loads((output / "dem-crop.json").read_text(encoding="utf-8"))
        manifest["region_ortho"] = prepare_region_ortho(cache, output, dem_metadata)
    (output / "pds-candidate.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {output / 'pds-candidate.json'}", flush=True)


if __name__ == "__main__":
    main()
