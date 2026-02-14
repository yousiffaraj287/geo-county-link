# geo-county-link

A lightweight Python geospatial data pipeline that queries public ArcGIS REST
services and performs automated spatial joins between waterbodies, counties,
and dam infrastructure datasets.

This repository contains a standalone implementation of a spatial analysis
workflow originally developed during a senior design project, redesigned here
as a reusable and container-friendly data pipeline.

---

## 🚀 Features
- Area-of-Interest (AOI) creation using bounding box + buffer distance
- Direct REST querying (no ArcPy required)
- Spatial clipping and joins using GeoPandas & Shapely
- Nearest-neighbor linking between datasets
- Automated CSV and GeoPackage outputs
- Designed for Docker / cloud deployment

---

## 🧭 Data Sources
- USGS National Hydrography Dataset (NHD)
- National Inventory of Dams (NID)
- U.S. Census County Boundaries

---

## ▶ Example Usage
```bash
python county_link.py --bbox -118.60 35.55 -118.35 35.75 --buffer "2 km"
---

## 📊 Example Output

Running the pipeline generates structured spatial datasets linking
waterbodies, counties, and dam infrastructure within a selected
area of interest.

Typical outputs include:

- Linked waterbody–county datasets (CSV)
- Dam proximity analysis tables
- GeoPackage (.gpkg) layers for GIS visualization
- Metadata logs describing query results

These outputs are designed for downstream analysis, mapping, and
decision-support workflows.
