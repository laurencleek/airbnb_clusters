# Airbnb pressure map (London)

Generates an interactive Leaflet/Folium HTML map that visualizes:
- **Borough pressure**: Airbnb listings as a percentage of borough housing stock.
- **Hotspot clusters**: spatial clusters of listings.
- **Low-pressure listings**: a capped set (top 1,500) of strong individual listings that are **not** in tourist-trap clusters.

The main output is a self-contained HTML file you can publish (e.g., GitHub Pages, personal website).

## Outputs

- `outputs/a_map.html` — website-ready map output (same content, convenient filename).
- `outputs/airbnb_checkin_clusters_map.html` — default generator output.

## Data files

Required:
- `data/london_boroughs.geojson` — London borough polygons.
- `data/borough_housing.xlsx` — housing stock by borough with columns:
  - `borough_code`
  - `Borough`
  - `Housing` (housing stock count)

Generated / required for regeneration:
- `outputs/london_airbnb_clusters_fixed.parquet` — clustered listings used by the map generator.

Optional / large:
- `data/listings.csv.gz` — raw Airbnb listings (not required if you already have the parquet above).

## Setup (Windows / PowerShell)

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Regenerate the map

Generate the website output:

```powershell
python scripts\map_clustering.py --output outputs\a_map.html
```

Generate the default output:

```powershell
python scripts\map_clustering.py
```

## Notes

- The borough overlay uses `Housing` from `data/borough_housing.xlsx` to compute:

  $$\text{pressure}(b) = \frac{\#\text{listings in } b}{\text{housing stock in } b} \times 100$$

- The fullscreen control is enabled; the left dashboard is forced to remain visible in fullscreen.

## License

See `LICENSE`.
