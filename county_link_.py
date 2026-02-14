"""
geo-county-link — Geospatial REST Data Pipeline

Author: Yousif Faraj
Description:
Standalone geospatial pipeline for querying public GIS services and
performing spatial joins between hydrography, county, and dam datasets.

Workflow Overview
-----------------
1. Defines an Area of Interest (AOI) via bounding box.
2. Applies an optional spatial buffer (e.g., "5 Kilometers" or "2 km").
3. Queries (via REST):
   - NHD Waterbodies (USGS Hydrography Service, Waterbody layer)
   - U.S. Detailed Counties (Census Service, Layer 2)
   - National Inventory of Dams (NID) points
4. Clips and joins:
   - NHD waterbodies ∩ counties
   - NID dams ∩ counties
   - NID dams ∩ nearest NHD waterbody (for viability metrics)
   - NHD waterbodies ∩ nearest NID dams (inner join, distance-based)
5. Outputs:
   - CSV for NHD waterbodies with county attributes
   - CSV for NID dams (metadata)
   - CSV "viability" table (dams + nearest waterbody + county)
   - CSV NHD–NID inner-join tables (raw + cleaned + dataset-template)
   - GeoPackage of spatial layers
   - JSON log summarizing query metadata / counts

Example:
    python county_link.py --bbox -118.60 35.55 -118.35 35.75 --buffer "2 km"
"""

from pathlib import Path
import argparse
import json
from typing import Optional

import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import (
    Polygon,
    MultiPolygon,
    Point,
    box,
    shape as shapely_shape,
)

# -------------------------------------------------------------
# Constants / configuration
# -------------------------------------------------------------

BASE = Path(__file__).parent
OUT_DIR = BASE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GPKG = OUT_DIR / "geo_outputs.gpkg"
LOG = OUT_DIR / "run_metadata.json"
NHD_CSV = OUT_DIR / "nhd_waterbodies_with_counties.csv"
NID_CSV = OUT_DIR / "nid_dams.csv"
VIABILITY_CSV = OUT_DIR / "nid_viability.csv"

# Public services
# NHD – Waterbody layer (polygons)
NHD_WATERBODY_URL = (
    "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/10"
)

# U.S. Detailed Counties (polygons)
COUNTIES_URL = (
    "https://services.arcgis.com/P3ePLMYs2RVChkJx/ArcGIS/rest/services/"
    "USA_County_Boundaries/FeatureServer/0"
)

# NID 2023 (points)
NID_URL = (
    "https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/"
    "NID_v1/FeatureServer/0"
)

DEFAULT_RECTS_LONLAT = [
    (-120.0, 34.5, -117.0, 36.5),  # Kern County box as an example
]
DEFAULT_BUFFER = "5 Kilometers"

# -------------------------------------------------------------
# Utility: parse distances like "5 Kilometers"
# -------------------------------------------------------------

def parse_buffer_distance(buf_str: str) -> float:
    """
    Parse a buffer distance like '5 Kilometers', '2 km', '500 m', or '1 mi'
    into meters. Never returns None.
    """
    if not buf_str:
        return 0.0

    s = str(buf_str).strip().lower()
    if not s:
        return 0.0

    parts = s.split()

    # Case 1: single token, e.g. "5" → assume km
    if len(parts) == 1:
        try:
            value = float(parts[0])
        except ValueError:
            raise ValueError(f"Could not parse buffer distance from '{buf_str}'")
        return value * 1000.0

    # Case 2: "number unit"
    num_part, unit_part = parts[0], parts[1]

    try:
        value = float(num_part)
    except ValueError:
        raise ValueError(f"Could not parse numeric buffer value from '{buf_str}'")

    if unit_part.startswith("km") or unit_part.startswith("kilo"):
        return value * 1000.0  # kilometers
    if unit_part.startswith("m"):
        return value           # meters
    if unit_part.startswith("mi"):
        return value * 1609.34 # miles

    raise ValueError(f"Unrecognized distance unit in '{buf_str}'")

# -------------------------------------------------------------
# AOI construction
# -------------------------------------------------------------

def build_aoi(rects_lonlat, bbox=None, buffer_str=None) -> gpd.GeoDataFrame:
    """
    Build an AOI GeoDataFrame from either:
      - a list of lon/lat rectangles, or
      - an explicit --bbox xmin ymin xmax ymax.

    Optionally buffer the AOI polygon by a distance string like
    "5 Kilometers" or "2 km".
    """
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError("--bbox must have four numbers: xmin ymin xmax ymax")
        xmin, ymin, xmax, ymax = bbox
        polys = [box(xmin, ymin, xmax, ymax)]
    else:
        polys = [box(*rect) for rect in rects_lonlat]

    aoi = gpd.GeoSeries(polys, crs="EPSG:4326").unary_union
    aoi_gdf = gpd.GeoDataFrame(geometry=[aoi], crs="EPSG:4326")

    if buffer_str:
        dist_m = parse_buffer_distance(buffer_str)
        if dist_m > 0:
            try:
                target_crs = aoi_gdf.estimate_utm_crs() or "EPSG:3857"
            except Exception:
                target_crs = "EPSG:3857"
            aoi_gdf = aoi_gdf.to_crs(target_crs).buffer(dist_m).to_crs(4326)
            aoi_gdf = gpd.GeoDataFrame(geometry=aoi_gdf, crs="EPSG:4326")

    return aoi_gdf

# -------------------------------------------------------------
# REST helpers
# -------------------------------------------------------------

def make_envelope_dict(aoi_gdf: gpd.GeoDataFrame) -> dict:
    """
    Build an envelope (xmin, ymin, xmax, ymax) for REST API queries.
    Always returns a valid bounding-box dictionary.
    """
    geom = aoi_gdf.geometry.unary_union
    xmin, ymin, xmax, ymax = geom.bounds
    return {
        "xmin": float(xmin),
        "ymin": float(ymin),
        "xmax": float(xmax),
        "ymax": float(ymax),
        "spatialReference": {"wkid": 4326},
    }

def _rest_query(url: str, env: dict) -> dict:
    """
    Generic helper to call an ArcGIS REST /query endpoint with an
    envelope geometry and return the JSON response.
    """
    params = {
        "where": "1=1",
        "geometry": json.dumps(env),
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": 4326,
        "f": "json",
        "resultRecordCount": 5000,
    }
    full_url = f"{url}/query"
    print(f"  -> REST request to {full_url}")
    resp = requests.get(full_url, params=params, timeout=60)
    resp.raise_for_status()
    try:
        data = resp.json()
    except Exception:
        data = json.loads(resp.text)

    if "error" in data:
        raise RuntimeError(f"REST error at {url}: {data['error']}")
    return data

# -------------------------------------------------------------
# Query functions (pure REST)
# -------------------------------------------------------------

def query_nhd(aoi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Query NHD waterbodies intersecting the AOI using a direct REST call.
    """
    env = make_envelope_dict(aoi_gdf)
    data = _rest_query(NHD_WATERBODY_URL, env)

    feats = data.get("features", [])
    if not feats:
        print("  -> NHD REST returned 0 features.")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    rows = []
    for f in feats:
        attrs = f.get("attributes", {}) or {}
        geom_json = f.get("geometry")
        if not geom_json:
            continue

        # NHD polygons come back as valid JSON for shapely.shape in this service
        try:
            geom = shapely_shape(geom_json)
        except Exception:
            # Fallback: ArcGIS-style rings
            rings = geom_json.get("rings") or []
            if not rings:
                continue
            polys = [Polygon(r) for r in rings if len(r) >= 3]
            if not polys:
                continue
            geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)

        attrs["geometry"] = geom
        rows.append(attrs)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    print(f"  -> NHD REST returned {len(gdf)} features.")
    return gdf

def query_counties(aoi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Query counties intersecting the AOI using a direct REST call.
    """
    env = make_envelope_dict(aoi_gdf)
    data = _rest_query(COUNTIES_URL, env)

    feats = data.get("features", [])
    if not feats:
        print("  -> Counties REST returned 0 features.")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    rows = []
    for f in feats:
        attrs = f.get("attributes", {}) or {}
        geom_json = f.get("geometry")
        if not geom_json:
            continue

        rings = geom_json.get("rings") or []
        if not rings:
            continue

        polys = [Polygon(r) for r in rings if len(r) >= 3]
        if not polys:
            continue
        geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)

        attrs["geometry"] = geom
        rows.append(attrs)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    print(f"  -> Counties REST returned {len(gdf)} features.")
    return gdf

def query_nid(aoi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Query NID dams intersecting the AOI using a direct REST call.
    Returns a GeoDataFrame of points in EPSG:4326.
    """
    env = make_envelope_dict(aoi_gdf)
    data = _rest_query(NID_URL, env)

    feats = data.get("features", [])
    if not feats:
        print("  -> NID REST returned 0 features.")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    rows = []
    for f in feats:
        attrs = f.get("attributes", {}) or {}
        geom_json = f.get("geometry")
        if not geom_json:
            continue
        x = geom_json.get("x")
        y = geom_json.get("y")
        if x is None or y is None:
            continue
        rows.append((*attrs.values(), x, y))

    if not rows:
        print("  -> NID REST: no valid geometries.")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    sample_attrs = feats[0].get("attributes", {}) or {}
    attr_names = list(sample_attrs.keys())
    df = pd.DataFrame(rows, columns=attr_names + ["x", "y"])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["x"], df["y"], crs="EPSG:4326"),
    )
    gdf.drop(columns=["x", "y"], inplace=True)
    print(f"  -> NID REST returned {len(gdf)} dams.")
    return gdf

# -------------------------------------------------------------
# Field-picking utilities
# -------------------------------------------------------------

def _pick_field_with_side(columns, prefer_side, *candidates):
    """
    prefer_side:
      None   -> any column
      'left' -> avoid obvious right-side columns (suffix '_right')
      'right'-> prefer columns that end with '_right'
    """
    if not columns:
        return None

    has_right_suffix = any(c.lower().endswith("_right") for c in columns)

    def side_ok(col_lower: str) -> bool:
        if prefer_side is None:
            return True
        if prefer_side == "left":
            return not col_lower.endswith("_right")
        if prefer_side == "right":
            if has_right_suffix:
                return col_lower.endswith("_right")
            return True
        return True

    lower_map = {c.lower(): c for c in columns}

    # Exact/prefix matches first
    for cand in candidates:
        base = cand.lower()
        if base in lower_map and side_ok(base):
            return lower_map[base]
        for low, orig in lower_map.items():
            if low.startswith(base + "_") and side_ok(low):
                return orig

    # Relaxed substring (ignore underscores)
    def norm(s: str) -> str:
        return s.replace("_", "")

    cand_norms = [norm(c.lower()) for c in candidates]
    for low, orig in lower_map.items():
        low_norm = norm(low)
        for cn in cand_norms:
            if cn in low_norm and side_ok(low):
                return orig

    return None

def pick_left_field(columns, *candidates):
    return _pick_field_with_side(columns, "left", *candidates)

def pick_right_field(columns, *candidates):
    return _pick_field_with_side(columns, "right", *candidates)

# -------------------------------------------------------------
# Outputs: NHD + county table
# -------------------------------------------------------------

def write_clean_outputs(joined_gdf):
    """
    Build a compact NHD + county CSV from the spatial join result.

    We don't try to be clever about "left/right" – we just look
    for column names that *look like* waterbody and county fields.
    """
    if joined_gdf.empty:
        print("Joined GeoDataFrame is empty; nothing to export.")
        return None

    joined_gdf = joined_gdf.reset_index(drop=True)
    cols = list(joined_gdf.columns)

    def find_col(candidates, exclude_substrings=None):
        """
        Return the first column whose uppercase name matches one of the
        candidate strings or contains it as a substring, while avoiding any
        of the exclude_substrings.
        """
        if exclude_substrings is None:
            exclude_substrings = []
        upper_map = {c: c.upper() for c in cols}

        for cand in candidates:
            cand_u = cand.upper()
            # exact match
            for col, cu in upper_map.items():
                if cu == cand_u and not any(ex in cu for ex in exclude_substrings):
                    return col
            # substring match
            for col, cu in upper_map.items():
                if cand_u in cu and not any(ex in cu for ex in exclude_substrings):
                    return col
        return None

    # --- waterbody fields (from NHD) ---
    wb_name = find_col(["GNIS_NAME", "WATERBODY_NAME", "NAME"])
    wb_ftype = find_col(["FTYPE", "FEATURE_TYPE"])
    wb_fcode = find_col(["FCODE"])
    wb_area  = find_col(["AREASQKM", "AREA_SQKM", "AREASQKM10", "AREA"])

    # --- county fields ---
    # Avoid GNIS_NAME etc. when looking for county name
    ct_name  = find_col(["NAME", "COUNTY_NAME"], exclude_substrings=["GNIS"])
    ct_geoid = find_col(
        ["GEOID", "GEOID20", "GEOID10", "CNTY_FIPS", "COUNTYFIPS", "FIPS", "FIPS_CODE"]
    )

    out = pd.DataFrame()
    out["Waterbody_Name"]      = joined_gdf[wb_name] if wb_name else ""
    out["NHD_FTYPE"]           = joined_gdf[wb_ftype] if wb_ftype else ""
    out["NHD_FCODE"]           = joined_gdf[wb_fcode] if wb_fcode else ""
    out["Waterbody_AreaSqKm"]  = joined_gdf[wb_area] if wb_area else ""
    out["County_Name"]         = joined_gdf[ct_name] if ct_name else ""
    out["County_GEOID"]        = joined_gdf[ct_geoid] if ct_geoid else ""

    out.to_csv(NHD_CSV, index=False)
    print(f"Wrote NHD+county CSV to {NHD_CSV}")

    # Save spatial layer to GeoPackage
    gdf_out = joined_gdf.copy()
    if gdf_out.crs is None:
        gdf_out = gdf_out.set_crs(epsg=4326, allow_override=True)

    if GPKG.exists():
        GPKG.unlink()
    gdf_out.to_file(GPKG, layer="nhd_with_counties", driver="GPKG")
    print(f"Wrote GeoPackage at {GPKG}")

    return out

# -------------------------------------------------------------
# NID outputs + viability + NHD–NID inner join
# -------------------------------------------------------------

def write_nid_outputs(nid_gdf: gpd.GeoDataFrame):
    if nid_gdf.empty:
        print("No NID dams in AOI; skipping NID exports.")
        return None

    try:
        df = nid_gdf.copy()
        if "geometry" in df.columns:
            df["geometry"] = df["geometry"].astype(str)
        if df.shape[1] == 0:
            df["feature_id"] = range(1, len(df) + 1)

        df.to_csv(NID_CSV, index=False)
        print(f"Wrote NID CSV to {NID_CSV}")

        if nid_gdf.crs is None:
            nid_gdf = nid_gdf.set_crs(epsg=4326, allow_override=True)
        nid_gdf.to_file(GPKG, layer="nid_dams", driver="GPKG")
        print(f"Appended NID dams layer to {GPKG}")
    except Exception as e:
        print(f"Error writing NID outputs: {e}")
        return None

    return df

def write_clean_inner_join_option_b(joined_gdf: gpd.GeoDataFrame):
    """
    Create a cleaned CSV (Option B) from the NHD–NID inner-join result.

    Columns:
      - Waterbody_Name
      - Waterbody_FType
      - Waterbody_AreaSqKm
      - Dam_Name
      - Dam_State
      - Dam_County
      - Dam_Purpose
      - Dam_Owner
      - Distance_m
    """
    if joined_gdf is None or joined_gdf.empty:
        print("Joined GeoDataFrame is empty; no cleaned NHD–NID CSV written.")
        return None

    joined_gdf = joined_gdf.drop_duplicates().reset_index(drop=True)
    cols = list(joined_gdf.columns)

    def find_col(candidates, exclude_substrings=None):
        if exclude_substrings is None:
            exclude_substrings = []
        upper_map = {c: c.upper() for c in cols}
        for cand in candidates:
            cand_u = cand.upper()
            # exact
            for col, cu in upper_map.items():
                if cu == cand_u and not any(ex in cu for ex in exclude_substrings):
                    return col
            # substring
            for col, cu in upper_map.items():
                if cand_u in cu and not any(ex in cu for ex in exclude_substrings):
                    return col
        return None

    out_df = pd.DataFrame()

    # --- waterbody fields (NHD) ---
    wb_name = find_col(["GNIS_NAME", "WATERBODY_NAME", "NAME"])
    wb_ftype = find_col(["FTYPE", "FEATURE_TYPE"])
    wb_area = find_col(["AREASQKM", "AREA_SQKM", "AREA"])

    out_df["Waterbody_Name"] = joined_gdf[wb_name] if wb_name else ""
    out_df["Waterbody_FType"] = joined_gdf[wb_ftype] if wb_ftype else ""
    out_df["Waterbody_AreaSqKm"] = joined_gdf[wb_area] if wb_area else ""

    # --- dam fields (NID/NTAD_Dams) ---
    dam_name = find_col(["DAM_NAME", "NAME", "DAMNM", "STRUCTNAME"])
    dam_state = find_col(["STATE", "ST_ABBREV"])
    dam_county = find_col(["COUNTY", "COUNTY_NAME"])
    dam_purpose = find_col(["PURPOSE", "PRIMARY_PURPOSE"])
    dam_owner = find_col(["OWNER", "OWNER_NAME", "OWNERS", "OWNER_TYPES"])

    out_df["Dam_Name"] = joined_gdf[dam_name] if dam_name else ""
    out_df["Dam_State"] = joined_gdf[dam_state] if dam_state else ""
    out_df["Dam_County"] = joined_gdf[dam_county] if dam_county else ""
    out_df["Dam_Purpose"] = joined_gdf[dam_purpose] if dam_purpose else ""
    out_df["Dam_Owner"] = joined_gdf[dam_owner] if dam_owner else ""

    # Distance (from sjoin_nearest)
    dist_col = find_col(["DISTANCE_M", "DIST_M", "DISTANCE"])
    out_df["Distance_m"] = joined_gdf[dist_col] if dist_col else ""

    # Remove duplicates on (Waterbody_Name, Dam_Name) so we don't repeat same pair
    dedup_cols = [c for c in ["Waterbody_Name", "Dam_Name"] if c in out_df.columns]
    if dedup_cols:
        out_df = out_df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    clean_path = OUT_DIR / "nhd_nid_inner_join_clean.csv"
    out_df.to_csv(clean_path, index=False)
    print(f"Wrote cleaned NHD–NID inner-join CSV to {clean_path}")
    return out_df

def write_nhd_nid_dataset_template(joined_gdf: gpd.GeoDataFrame):
    """
    Map the NHD–NID inner-join result into the generic senior-design
    dataset template structure (Dataset Inventory.xlsx → Dataset Template).
    """
    if joined_gdf is None or joined_gdf.empty:
        print("Joined GeoDataFrame is empty; skipping dataset-template CSV.")
        return None

    if joined_gdf.crs is None:
        joined_gdf = joined_gdf.set_crs(epsg=4326, allow_override=True)
    else:
        joined_gdf = joined_gdf.to_crs(4326)

    cols_template = [
        "Name of water body",
        "Longitude",
        "Latitude",
        "Water ownership information",
        "Water owner",
        "Electrical utility company service territory",
        "Water Agency Districts service territory",
        "ISO Service territory",
        "Regional Electricity Costs",
        "Regional Power purchase agreement (PPA) averages",
        "Census areas",
        "Tribal lands",
        "Land Parcel Data",
        "Sensitive habitats",
        "Water bird migratory areas",
        "Fish Presence and type",
        "Other Protected/inaccessible/otherwise unviable lands",
        "Border ID",
        "Type ID",
        "Recreation",
        "Bathymetry Data",
        "Dimensional information",
        "Dry-up status",
        "Max and minimum water level",
        "Algae presence",
        "Water Temperature",
    ]

    out_df = pd.DataFrame(columns=cols_template)

    # Name of water body
    if "gnis_name" in joined_gdf.columns:
        out_df["Name of water body"] = joined_gdf["gnis_name"].astype(str)
    elif "GNIS_NAME" in joined_gdf.columns:
        out_df["Name of water body"] = joined_gdf["GNIS_NAME"].astype(str)
    else:
        out_df["Name of water body"] = ""

    # Lon/Lat from waterbody centroid
    try:
        centroids = joined_gdf.geometry.centroid
        out_df["Longitude"] = centroids.x
        out_df["Latitude"] = centroids.y
    except Exception:
        out_df["Longitude"] = ""
        out_df["Latitude"] = ""

    # Water owner / ownership info from dam owner fields
    owner_series = None
    for cand in ["owner", "ownerNames", "OWNER", "OWNER_TYPES", "PRIMARY_OWNER_TYPE"]:
        if cand in joined_gdf.columns:
            owner_series = joined_gdf[cand].astype(str)
            break
    if owner_series is None:
        owner_series = pd.Series([""] * len(joined_gdf))

    out_df["Water owner"] = owner_series
    out_df["Water ownership information"] = owner_series

    out_df = out_df.drop_duplicates(
        subset=["Name of water body", "Longitude", "Latitude", "Water owner"]
    ).reset_index(drop=True)

    template_path = OUT_DIR / "nhd_nid_dataset_template.csv"
    out_df.to_csv(template_path, index=False)
    print(f"Wrote NHD–NID dataset-template CSV to {template_path}")
    return out_df

def build_nhd_nid_inner_join(
    nhd_clip: gpd.GeoDataFrame,
    nid_clip: gpd.GeoDataFrame,
    max_distance_km: float = 10.0,
) -> Optional[gpd.GeoDataFrame]:
    """
    Inner-join style nearest-neighbor link between NHD waterbodies and NID dams.
    """
    if nhd_clip.empty or nid_clip.empty:
        print("Cannot build NHD–NID join; one of the inputs is empty.")
        return None

    try:
        target_crs = nhd_clip.estimate_utm_crs() or "EPSG:3857"
    except Exception:
        target_crs = "EPSG:3857"

    nhd_proj = nhd_clip.to_crs(target_crs)
    nid_proj = nid_clip.to_crs(target_crs)

    max_dist_m = max_distance_km * 1000.0

    joined = gpd.sjoin_nearest(
        nhd_proj,
        nid_proj,
        how="inner",
        max_distance=max_dist_m,
        distance_col="distance_m",
    )

    if joined.empty:
        print("No NHD–NID pairs found within the distance threshold.")
        return None

    joined = joined.to_crs(4326)
    out_csv = OUT_DIR / "nhd_nid_inner_join.csv"
    joined.to_csv(out_csv, index=False)
    print(f"Wrote raw NHD–NID inner-join CSV to {out_csv}")
    return joined

def write_viability_csv(
    nid_clip: gpd.GeoDataFrame,
    nhd_clip: gpd.GeoDataFrame,
    ct_clip: gpd.GeoDataFrame,
):
    """
    Build a "viability" table combining:
    - Key NID attributes (dam metadata)
    - NHD attributes (nearest waterbody within 10 km)
    - County FIPS / GEOID & name
    """
    if nid_clip.empty:
        print("No NID dams in AOI; skipping viability CSV.")
        return None

    df = nid_clip.copy()
    if df.crs is None:
        df = df.set_crs(epsg=4326, allow_override=True)

    # 1) Join each dam to its county (point-in-polygon)
    if not ct_clip.empty:
        ct_clip = ct_clip.to_crs(df.crs)
        dam_in_ct = gpd.sjoin(
            df,
            ct_clip,
            how="left",
            predicate="intersects",
        )
        # Avoid collisions with later sjoin_nearest
        if "index_right" in dam_in_ct.columns:
            dam_in_ct = dam_in_ct.rename(columns={"index_right": "county_index"})
    else:
        dam_in_ct = df.copy()

    # 2) Join dams to nearest NHD waterbody within 10 km
    if nhd_clip.empty:
        wb_gnis_vals = ["" for _ in range(len(dam_in_ct))]
        wb_type_vals = ["" for _ in range(len(dam_in_ct))]
        wb_area_vals = ["" for _ in range(len(dam_in_ct))]
    else:
        try:
            target_crs = nhd_clip.estimate_utm_crs() or "EPSG:3857"
        except Exception:
            target_crs = "EPSG:3857"

        nhd_proj = nhd_clip.to_crs(target_crs)
        dams_proj = dam_in_ct.to_crs(target_crs)

        # Clean up index_* columns so GeoPandas doesn't complain
        for col in ("index_right", "index_left"):
            if col in nhd_proj.columns:
                nhd_proj = nhd_proj.rename(columns={col: f"{col}_nhd"})
            if col in dams_proj.columns:
                dams_proj = dams_proj.rename(columns={col: f"{col}_nid"})

        joined = gpd.sjoin_nearest(
            dams_proj,
            nhd_proj,
            how="left",
            max_distance=10000.0,
            distance_col="dist_m",
        )
        joined = joined.to_crs(4326)

        cols_joined = list(joined.columns)

        # Prefer GNIS_NAME / WATERBODY_NAME, NOT GNIS_ID
        def find_wb_name_field(columns):
            # exact preferred names
            for pref in ["GNIS_NAME", "WATERBODY_NAME"]:
                for c in columns:
                    if c.upper() == pref:
                        return c
            # any column that has both GNIS and NAME in it
            for c in columns:
                cu = c.upper()
                if "GNIS" in cu and "NAME" in cu:
                    return c
            return None

        def find_wb_field(columns, keywords):
            for c in columns:
                cu = c.upper()
                if any(k in cu for k in keywords):
                    return c
            return None

        wb_name_field = find_wb_name_field(cols_joined)
        wb_type_field = find_wb_field(cols_joined, ["FTYPE", "FEATURE_TYPE"])
        wb_area_field = find_wb_field(cols_joined, ["AREASQKM", "AREA_SQKM"])

        wb_gnis_vals = []
        wb_type_vals = []
        wb_area_vals = []

        for _, row in joined.iterrows():
            wb_gnis_vals.append(row.get(wb_name_field, "") if wb_name_field else "")
            wb_type_vals.append(row.get(wb_type_field, "") if wb_type_field else "")
            wb_area_vals.append(row.get(wb_area_field, "") if wb_area_field else "")

        dam_in_ct = joined

    # Fallback if lists didn't get filled above
    if not wb_gnis_vals:
        wb_gnis_vals = ["" for _ in range(len(dam_in_ct))]
        wb_type_vals = ["" for _ in range(len(dam_in_ct))]
        wb_area_vals = ["" for _ in range(len(dam_in_ct))]

    dam_in_ct["__wb_gnis"] = wb_gnis_vals
    dam_in_ct["__wb_type"] = wb_type_vals
    dam_in_ct["__wb_area"] = wb_area_vals

    cols = list(dam_in_ct.columns)

    def pf_left(*names):
        return pick_left_field(cols, *names)

    # We’ll pick county fields ourselves (not pf_right) so we don’t miss FIPS/GEOID
    def find_ct_field(candidates):
        """
        Look for any column whose name matches or contains one of the
        candidate tokens (e.g., GEOID, FIPS, COUNTYFIPS, etc.).
        Works whether or not '_right' suffixes are present.
        """
        cand_norms = [c.upper().replace("_", "") for c in candidates]
        for col in cols:
            cn = col.upper().replace("_", "")
            for cand in cand_norms:
                if cand == cn or cand in cn:
                    return col
        return None

    # NID dam fields (left side)
    fld_name        = pf_left("NAME")
    fld_owner_types = pf_left("OWNER_TYPES", "OWNER_TYPE")
    fld_primary_own = pf_left("PRIMARY_OWNER_TYPE")
    fld_fed_id      = pf_left("FEDERAL_ID")
    fld_river       = pf_left("RIVER_OR_STREAM", "RIVER")
    fld_lat         = pf_left("LATITUDE", "LAT")
    fld_lon         = pf_left("LONGITUDE", "LON", "LONG")
    fld_nid_county  = pf_left("COUNTYS", "COUNTY")
    fld_nid_state   = pf_left("STATE")

    # County fields from spatial join (any side)
    fld_ct_name = find_ct_field(["NAME", "COUNTY_NAME"])
    fld_ct_geoid = find_ct_field(
        ["GEOID", "GEOID20", "GEOID10", "CNTY_FIPS", "COUNTYFIPS", "FIPS", "FIPS_CODE"]
    )

    out = pd.DataFrame()
    out["Dam_Name"] = dam_in_ct[fld_name] if fld_name else ""
    out["Dam_River"] = dam_in_ct[fld_river] if fld_river else ""
    out["Dam_Latitude"] = dam_in_ct[fld_lat] if fld_lat else ""
    out["Dam_Longitude"] = dam_in_ct[fld_lon] if fld_lon else ""
    out["Dam_County_From_NID"] = dam_in_ct[fld_nid_county] if fld_nid_county else ""
    out["Dam_State"] = dam_in_ct[fld_nid_state] if fld_nid_state else ""
    out["Dam_Owner_Types"] = dam_in_ct[fld_owner_types] if fld_owner_types else ""
    out["Dam_Primary_Owner_Type"] = dam_in_ct[fld_primary_own] if fld_primary_own else ""
    out["Dam_Federal_ID"] = dam_in_ct[fld_fed_id] if fld_fed_id else ""

    out["County_Name"] = dam_in_ct[fld_ct_name] if fld_ct_name else ""
    out["County_GEOID"] = dam_in_ct[fld_ct_geoid] if fld_ct_geoid else ""

    out["Waterbody_GNIS_Name"] = dam_in_ct["__wb_gnis"]
    out["Waterbody_Type"] = dam_in_ct["__wb_type"]
    out["Waterbody_AreaSqKm"] = dam_in_ct["__wb_area"]

    desired_order = [
        "Dam_Name",
        "Dam_River",
        "Dam_Latitude",
        "Dam_Longitude",
        "Dam_County_From_NID",
        "Dam_State",
        "Dam_Owner_Types",
        "Dam_Primary_Owner_Type",
        "Dam_Federal_ID",
        "Waterbody_GNIS_Name",
        "Waterbody_Type",
        "Waterbody_AreaSqKm",
        "County_Name",
        "County_GEOID",
    ]
    for col in desired_order:
        if col not in out.columns:
            out[col] = ""

    out = out[desired_order]
    out.to_csv(VIABILITY_CSV, index=False)
    print(f"Wrote viability CSV file to {VIABILITY_CSV}")

    return out
def postprocess_viability_from_nhd():
    """
    After write_viability_csv runs, fix county info in the viability CSV
    using the NHD+county table as the single source of truth.

    - Set County_Name and County_GEOID to match nhd_waterbodies_with_counties.csv
    - Drop Dam_County_From_NID (it's redundant / confusing)
    """
    try:
        if not NHD_CSV.exists() or not VIABILITY_CSV.exists():
            return

        nhd_df = pd.read_csv(NHD_CSV)
        viab_df = pd.read_csv(VIABILITY_CSV)

        if nhd_df.empty or viab_df.empty:
            return

        # Take the first row – in our use case AOI is a single county
        county_name = nhd_df["County_Name"].iloc[0] if "County_Name" in nhd_df.columns else ""
        county_geoid = nhd_df["County_GEOID"].iloc[0] if "County_GEOID" in nhd_df.columns else ""

        # Overwrite county fields in viability table
        viab_df["County_Name"] = county_name
        viab_df["County_GEOID"] = county_geoid

        # Remove noisy NID county text column (we already have cleaned county info)
        if "Dam_County_From_NID" in viab_df.columns:
            viab_df = viab_df.drop(columns=["Dam_County_From_NID"])

        viab_df.to_csv(VIABILITY_CSV, index=False)
        print("Post-processed viability CSV with county info from NHD table.")
    except Exception as e:
        print(f"Warning: could not post-process viability CSV: {e}")

def write_combined_excel(
    nhd_df: Optional[pd.DataFrame],
    viability_df: Optional[pd.DataFrame],
    nid_df: Optional[pd.DataFrame],
):
    print("Building combined Excel workbook (NHD, viability, NID).")

    if (
        (nhd_df is None or nhd_df.empty)
        and (viability_df is None or viability_df.empty)
        and (nid_df is None or nid_df.empty)
    ):
        print("No data available for combined Excel; skipping.")
        return

    xlsx_path = OUT_DIR / "geo_combined.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        if nhd_df is not None and not nhd_df.empty:
            nhd_df.to_excel(writer, sheet_name="NHD_Waterbodies", index=False)
        if viability_df is not None and not viability_df.empty:
            viability_df.to_excel(writer, sheet_name="NID_Viability", index=False)
        if nid_df is not None and not nid_df.empty:
            nid_df.to_excel(writer, sheet_name="NID_Dams", index=False)

    print(f"Wrote combined Excel workbook to {xlsx_path}")

# -------------------------------------------------------------
# CLI arguments
# -------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Geospatial REST pipeline for NHD/NID spatial analysis")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        help="Bounding box: xmin ymin xmax ymax (in lon/lat).",
    )
    parser.add_argument(
        "--buffer",
        type=str,
        help="Buffer distance like '5 Kilometers' applied to the AOI.",
    )
    return parser.parse_args()

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():
    args = parse_args()
    info = {
        "ok": False,
        "nhd_count": 0,
        "county_count": 0,
        "nid_count": 0,
        "reason": None,
    }

    aoi_gdf = build_aoi(
        rects_lonlat=DEFAULT_RECTS_LONLAT,
        bbox=args.bbox,
        buffer_str=args.buffer or DEFAULT_BUFFER,
    )
    print("AOI ready")

    # NHD
    print("Querying NHD waterbodies in AOI .")
    try:
        nhd_gdf = query_nhd(aoi_gdf)
        info["nhd_count"] = len(nhd_gdf)
        print(f"  NHD features returned: {info['nhd_count']}")
    except Exception as e:
        info["reason"] = f"NHD query failed: {type(e).__name__}: {e}"
        print("Error:", info["reason"])
        LOG.write_text(json.dumps(info, indent=2))
        return

    if info["nhd_count"] == 0:
        info["reason"] = "No NHD waterbodies in AOI; try a different bbox/buffer."
        print(info["reason"])
        LOG.write_text(json.dumps(info, indent=2))
        return

    # Counties
    print("Querying counties in AOI .")
    try:
        ct_gdf = query_counties(aoi_gdf)
        info["county_count"] = len(ct_gdf)
        print(f"  County features returned: {info['county_count']}")
    except Exception as e:
        info["reason"] = f"County query failed: {type(e).__name__}: {e}"
        print("Error:", info["reason"])
        LOG.write_text(json.dumps(info, indent=2))
        return

    # NID
    nid_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    print("Querying NID dams in AOI .")
    try:
        nid_gdf = query_nid(aoi_gdf)
        info["nid_count"] = len(nid_gdf)
        print(f"  NID features returned: {info['nid_count']}")
    except Exception as e:
        info["reason"] = f"NID query failed: {type(e).__name__}: {e}"
        print("Error:", info["reason"])
        LOG.write_text(json.dumps(info, indent=2, default=str))
        return

    # NHD + counties join
    if not ct_gdf.empty:
        nhd_clip = gpd.clip(nhd_gdf, aoi_gdf)
        ct_clip = gpd.clip(ct_gdf, aoi_gdf)
        joined = gpd.sjoin(nhd_clip, ct_clip, how="left", predicate="intersects")
    else:
        nhd_clip = gpd.clip(nhd_gdf, aoi_gdf)
        ct_clip = ct_gdf
        joined = nhd_clip

    print(f"Joined rows: {len(joined)}")
    nhd_table = write_clean_outputs(joined)

    nid_table: Optional[pd.DataFrame] = None
    viability_table: Optional[pd.DataFrame] = None

    if not nid_gdf.empty:
        nid_clip = gpd.clip(nid_gdf, aoi_gdf)
        print(f"NID dams in AOI: {len(nid_clip)}")
        nid_table = write_nid_outputs(nid_clip)

        # NHD–NID inner join & template
        nhd_nid_join = build_nhd_nid_inner_join(
            nhd_clip,
            nid_clip,
            max_distance_km=10.0,
        )
        nhd_nid_clean = write_clean_inner_join_option_b(nhd_nid_join)
        template_table = write_nhd_nid_dataset_template(nhd_nid_join)

        # Viability CSV (dam-centric summary)
        viability_table = write_viability_csv(nid_clip, nhd_clip, ct_clip)

        # Fix County_Name / County_GEOID & drop Dam_County_From_NID
        postprocess_viability_from_nhd()
    else:
        print("No NID dams found or NID query failed; no NID exports created.")
        nhd_nid_join = None
        nhd_nid_clean = None
        template_table = None

    write_combined_excel(nhd_table, viability_table, nid_table)

    info["ok"] = True
    LOG.write_text(json.dumps(info, indent=2, default=str))
    print(f"Log written to {LOG}")

if __name__ == "__main__":
    main()
