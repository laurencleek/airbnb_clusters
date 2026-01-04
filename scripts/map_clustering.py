from __future__ import annotations

from pathlib import Path

import numpy as np
import json
import html
import pandas as pd
import folium
from folium import plugins
import branca.colormap as bcm
from matplotlib import cm
from matplotlib import colors as mcolors
from scipy.spatial import ConvexHull
import csv


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"

LISTINGS_PATH = OUTPUTS_DIR / "london_airbnb_clusters_fixed.parquet"
BOROUGHS_GEOJSON_PATH = DATA_DIR / "london_boroughs.geojson"  # optional
HOUSING_PATH_XLSX = DATA_DIR / "borough_housing.xlsx"
HOUSING_PATH_2020 = DATA_DIR / "local-authority-housing-stock-borough.csv"
HOUSING_PATH_LEGACY = DATA_DIR / "borough_housing_stock.csv"

OUTPUT_HTML = OUTPUTS_DIR / "airbnb_checkin_clusters_map.html"


MAX_LOW_PRESSURE_LISTINGS = 1500


def _read_housing_stock_2020(path: Path) -> pd.DataFrame:
    """Read the local authority housing stock CSV.

    The source file may contain rows that are accidentally wrapped as a single quoted field
    (everything ends up in the first column, with the other columns as NaN). When detected,
    this repairs the file by parsing those rows manually.
    """

    df = pd.read_csv(path)
    if "Area" not in df.columns or "Number of properties 2020" not in df.columns:
        return df

    # Heuristic: if most rows have Area missing, the file is likely quote-wrapped per-row.
    if float(df["Area"].isna().mean()) < 0.2:
        return df

    with open(path, "r", encoding="utf-8", newline="") as f:
        lines = f.read().splitlines()
    if not lines:
        return df

    header = next(csv.reader([lines[0]]))
    fixed_rows: list[list[str]] = []
    for raw in lines[1:]:
        line = raw.strip()
        if not line:
            continue
        # If the entire row is wrapped in quotes, unwrap it and unescape doubled quotes.
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1].replace('""', '"')
        fixed_rows.append(next(csv.reader([line])))

    fixed = pd.DataFrame(fixed_rows, columns=header)
    return fixed


def _read_housing_stock_xlsx(path: Path) -> pd.DataFrame:
    """Read borough housing stock from the Excel file.

    Expected columns: borough_code, Borough, Housing
    We only need Borough and Housing.
    """

    df = pd.read_excel(path)
    if "Borough" not in df.columns or "Housing" not in df.columns:
        raise ValueError(
            "borough_housing.xlsx must contain columns 'Borough' and 'Housing'"
        )

    out = df[["Borough", "Housing"]].rename(columns={"Borough": "borough", "Housing": "total_dwellings"}).copy()
    out["borough"] = out["borough"].astype(str).str.strip()
    out["total_dwellings"] = pd.to_numeric(out["total_dwellings"], errors="coerce")
    return out


def _hex_palette(n: int) -> list[str]:
    # Matplotlib >=3.7: prefer colormaps API
    try:
        from matplotlib import colormaps

        cmap = colormaps.get_cmap("tab20" if n <= 20 else "hsv")
        return [mcolors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]
    except Exception:
        cmap = cm.get_cmap("tab20", max(n, 1)) if n <= 20 else cm.get_cmap("hsv", max(n, 1))
        return [mcolors.to_hex(cmap(i)) for i in range(n)]


def _cluster_label(size: int) -> str:
    if size > 5000:
        return "Megacluster"
    if size > 1500:
        return "Large cluster"
    if size > 500:
        return "Medium cluster"
    return "Local pocket"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Great-circle distance (km)
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def main(output_html: Path | None = None) -> None:
    if not LISTINGS_PATH.exists():
        raise FileNotFoundError(f"Missing {LISTINGS_PATH}.")

    print(f"Loading listings: {LISTINGS_PATH}")
    lock = pd.read_parquet(LISTINGS_PATH)
    required = {"latitude", "longitude", "cluster_id"}
    missing = required - set(lock.columns)
    if missing:
        raise ValueError(f"Listings file missing columns: {sorted(missing)}")

    housing = None
    if HOUSING_PATH_XLSX.exists():
        housing = _read_housing_stock_xlsx(HOUSING_PATH_XLSX)
    elif HOUSING_PATH_2020.exists():
        housing = _read_housing_stock_2020(HOUSING_PATH_2020)
    elif HOUSING_PATH_LEGACY.exists():
        housing = pd.read_csv(HOUSING_PATH_LEGACY)

    borough_col = "borough" if "borough" in lock.columns else "neighbourhood_cleansed" if "neighbourhood_cleansed" in lock.columns else None

    # Parse price into numeric if present (e.g. "$123.00")
    if "price" in lock.columns:
        price_numeric = (
            lock["price"]
            .astype(str)
            .str.replace(r"[^0-9.]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )
        lock = lock.copy()
        lock["price_numeric"] = price_numeric

    # --- Compute cluster stats ---
    print("Computing cluster stats...")
    cluster_sizes = lock.loc[lock["cluster_id"] != -1, "cluster_id"].value_counts()
    cluster_ids = cluster_sizes.index.sort_values().tolist()
    palette = _hex_palette(len(cluster_ids))
    cluster_color = {cid: palette[i] for i, cid in enumerate(cluster_ids)}

    # Ensure cluster 1 and 2 are clearly distinct (when present)
    if 1 in cluster_color:
        cluster_color[1] = "#2563eb"  # blue
    if 2 in cluster_color:
        cluster_color[2] = "#f97316"  # orange

    def color_for(cid: int) -> str:
        return "#bdbdbd" if cid == -1 else cluster_color.get(cid, "#333333")

    # --- Map ---
    print("Building Folium map...")
    m = folium.Map(
        location=[51.509865, -0.118092],
        zoom_start=11,
        tiles="CartoDB Positron",
        control_scale=True,
        prefer_canvas=True,
    )

    # Ensure the map has a meaningful browser tab title (the sidebar title is separate).
    m.get_root().header.add_child(folium.Element("<title>Airbnb pressure map</title>"))

    # Minimal, professional controls
    plugins.Fullscreen(position="topright").add_to(m)
    plugins.LocateControl(position="topright", locateOptions={"enableHighAccuracy": True}).add_to(m)

    # --- Sidebar + CSS (collapsible, clean) ---
    total = int(len(lock))
    clustered = int((lock["cluster_id"] != -1).sum())
    n_clusters = int(len(cluster_sizes))

    top_clusters = cluster_sizes.head(5)

    top_clusters_html = "".join(
        f"<tr><td style='padding:4px 6px;'><span style='display:inline-block;width:10px;height:10px;border-radius:2px;background:{color_for(int(cid))};margin-right:6px;'></span>Cluster {int(cid)}</td>"
        f"<td style='padding:4px 6px;text-align:right;'>{int(sz):,}</td>"
        f"<td style='padding:4px 6px;color:#666;'>{_cluster_label(int(sz))}</td></tr>"
        for cid, sz in top_clusters.items()
    )

    def _iter_coords(geom):
        """Yield (lon, lat) from GeoJSON geometry coordinates."""
        if not geom:
            return
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if not coords:
            return
        if gtype == "Polygon":
            for ring in coords:
                for lon, lat in ring:
                    yield float(lon), float(lat)
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    for lon, lat in ring:
                        yield float(lon), float(lat)

    def _bounds_for_feature(feat) -> tuple[float, float, float, float] | None:
        xs: list[float] = []
        ys: list[float] = []
        for x, y in _iter_coords(feat.get("geometry")):
            xs.append(x)
            ys.append(y)
        if not xs:
            return None
        return min(xs), min(ys), max(xs), max(ys)

    # Borough dropdown will be injected only if we can compute bounds.
    borough_dropdown_html = ""
    borough_bounds_script = ""
    borough_geojson: dict | None = None
    if BOROUGHS_GEOJSON_PATH.exists():
        try:
            with open(BOROUGHS_GEOJSON_PATH, "r", encoding="utf-8") as f:
                borough_geojson = json.load(f)

            bounds_map: dict[str, list[list[float]]] = {}
            all_x: list[float] = []
            all_y: list[float] = []
            for feat in (borough_geojson.get("features") or []):
                name = str((feat.get("properties") or {}).get("name") or "").strip()
                b = _bounds_for_feature(feat)
                if not name or b is None:
                    continue
                minx, miny, maxx, maxy = b
                bounds_map[name] = [[float(miny), float(minx)], [float(maxy), float(maxx)]]
                all_x.extend([minx, maxx])
                all_y.extend([miny, maxy])

            if all_x and all_y:
                bounds_map["All London"] = [[float(min(all_y)), float(min(all_x))], [float(max(all_y)), float(max(all_x))]]

            if bounds_map:
                options = ["All London"] + sorted([k for k in bounds_map.keys() if k != "All London"])
                options_html = "".join(f"<option value='{o}'>{o}</option>" for o in options)
                borough_dropdown_html = f"""
                  <div class="section">
                    <h3>Explore By Borough</h3>
                    <select id="boroughSelect" style="width:100%;padding:8px 10px;border-radius:10px;border:1px solid rgba(0,0,0,0.12);background:#fff;font-size:12px;">
                      {options_html}
                    </select>
                    <div style="margin-top:6px;font-size:12px;color:var(--muted);">Zoom to a borough boundary (map stays interactive).</div>
                  </div>
                """
                borough_bounds_script = f"<script>window.__BOROUGH_BOUNDS__ = {json.dumps(bounds_map)};</script>"
        except Exception:
            borough_geojson = None
            borough_dropdown_html = ""
            borough_bounds_script = ""

    sidebar = f"""
    <style>
      :root {{ --panel-bg: rgba(255,255,255,0.97); --panel-border: rgba(0,0,0,0.08); --text: #1f2937; --muted: #6b7280; }}
            #panel {{ position: fixed; top: 24px; left: 24px; width: 340px; z-index: 2147483647; font-family: 'Segoe UI', system-ui, -apple-system, Arial, sans-serif; color: var(--text); }}
            /* Leaflet fullscreen mode sometimes changes stacking contexts; force the dashboard to remain visible. */
            body.leaflet-fullscreen-on #panel {{ display: block !important; visibility: visible !important; opacity: 1 !important; z-index: 2147483647 !important; }}
      #panel .card {{ background: var(--panel-bg); border: 1px solid var(--panel-border); border-radius: 14px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); overflow: hidden; }}
      #panel .hdr {{ padding: 14px 16px 10px 16px; border-bottom: 1px solid rgba(0,0,0,0.06); display:flex; align-items:center; justify-content:space-between; gap: 12px; }}
      #panel .title {{ font-size: 16px; font-weight: 700; line-height: 1.2; }}
        #panel .btnRow {{ display:flex; gap: 8px; align-items:center; }}
        #panel .btn {{ border: 1px solid rgba(0,0,0,0.14); background: #111827; color: #fff; border-radius: 12px; padding: 8px 12px; font-size: 12px; font-weight: 700; cursor: pointer; box-shadow: 0 6px 16px rgba(0,0,0,0.12); }}
        #panel .btn:hover {{ filter: brightness(1.05); }}
        #panel .btnHelp {{ width: 34px; padding: 8px 0; text-align:center; }}
            #panel .body {{ padding: 12px 16px 14px 16px; max-height: calc(100vh - 120px); overflow: auto; }}
      #panel .kpis {{ display:grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px; }}
      #panel .kpi {{ border: 1px solid rgba(0,0,0,0.06); border-radius: 12px; padding: 10px; background: rgba(250,250,250,0.6); }}
      #panel .kpi .n {{ font-size: 18px; font-weight: 800; }}
      #panel .kpi .l {{ font-size: 12px; color: var(--muted); }}
      #panel .section {{ margin-top: 10px; }}
      #panel .section h3 {{ font-size: 12px; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); margin: 10px 0 6px 0; }}
      #panel table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
      #panel .legend {{ display:flex; gap:10px; flex-wrap:wrap; font-size:12px; color: var(--muted); margin-top: 8px; }}
      #panel .chip {{ display:flex; align-items:center; gap:6px; }}
      #panel .sw {{ width: 10px; height: 10px; border-radius: 3px; display:inline-block; }}
        #panel .layers {{ border: 1px solid rgba(0,0,0,0.06); border-radius: 12px; background: rgba(250,250,250,0.6); padding: 10px; }}
        #panel .layers label {{ display:flex; align-items:flex-start; gap:10px; margin: 8px 0; cursor: pointer; }}
        #panel .layers input {{ width: 16px; height: 16px; margin-top: 1px; }}
        #panel .layers .lname {{ font-size: 12px; color: var(--text); font-weight: 700; line-height: 1.2; }}
        #panel .layers .lnote {{ font-size: 12px; color: var(--muted); line-height: 1.25; margin-top: 2px; }}
            #panel .credit {{ margin-top: 10px; font-size: 12px; color: var(--muted); display:flex; justify-content:space-between; gap: 10px; }}
            #panel .credit b {{ color: #111827; }}
            #panel .infoIcon {{ display:inline-flex; align-items:center; justify-content:center; width: 18px; height: 18px; border-radius: 999px; border: 1px solid rgba(0,0,0,0.16); color: #111827; background: rgba(255,255,255,0.9); font-size: 12px; font-weight: 800; cursor: help; }}

            @media (max-width: 900px) {{
                #panel {{ left: 10px; top: 10px; width: 320px; }}
                #panel .body {{ max-height: calc(100vh - 96px); }}
            }}

            /* Borough legend is moved into the left panel via JS */
    </style>
    {borough_bounds_script}
    <div id="panel">
      <div class="card">
        <div class="hdr">
          <div>
                                                <div class="title">Airbnb pressure map</div>
          </div>
                                        <div class="btnRow">
                                            <button class="btn btnHelp" id="helpBtn" title="Help">?</button>
                                            <button class="btn" onclick="var b=document.getElementById('panelBody'); b.style.display = (b.style.display==='none'?'block':'none');">SHOW / HIDE</button>
                                        </div>
        </div>
        <div class="body" id="panelBody">
          <div class="kpis">
            <div class="kpi"><div class="n">{total:,}</div><div class="l">Listings</div></div>
            <div class="kpi"><div class="n">{n_clusters:,}</div><div class="l">Clusters found</div></div>
            <div class="kpi"><div class="n">{clustered:,}</div><div class="l">Clustered points</div></div>
          </div>
          <div class="section">
            <h3>Top Clusters</h3>
            <table>
              <thead><tr><th style='text-align:left;padding:4px 6px;'>Cluster</th><th style='text-align:right;padding:4px 6px;'>Listings</th><th style='text-align:left;padding:4px 6px;'>Type</th></tr></thead>
              <tbody>
                {top_clusters_html}
              </tbody>
            </table>
          </div>

          <div class="legend">
            <div class="chip"><span class="sw" style="background:#111827"></span>Cluster footprint</div>
            <div class="chip"><span class="sw" style="background:#111827"></span>Cluster centroid</div>
                                                <div class="chip"><span class="sw" style="background:#10b981"></span>Low-Pressure Listings</div>
          </div>

                    <div style="margin-top:10px;font-size:12px;color:var(--muted);display:flex;align-items:center;gap:8px;">
                        <span>About the layers</span>
                        <span class="infoIcon" title="Designed for clarity: footprints + centroids by default. Turn on sampled listing points only if you need raw texture.">i</span>
                    </div>

                    <div class="section">
                        <h3>Layers</h3>
                        <div class="layers">
                            <label>
                                <input id="toggleFootprints" type="checkbox" checked />
                                <div>
                                    <div class="lname">Cluster footprints</div>
                                    <div class="lnote">Simplified hull outlines (readable at city scale).</div>
                                </div>
                            </label>
                            <label>
                                <input id="toggleCentroids" type="checkbox" checked />
                                <div>
                                    <div class="lname">Cluster centroids</div>
                                    <div class="lnote">Bubble markers sized by cluster count.</div>
                                </div>
                            </label>
                            <label>
                                <input id="toggleBorough" type="checkbox" checked />
                                <div>
                                    <div class="lname">Borough pressure (normalized)</div>
                                    <div class="lnote">Listings as % of housing stock (if available).</div>
                                </div>
                            </label>
                            <label>
                                <input id="toggleSample" type="checkbox" />
                                <div>
                                    <div class="lname">Sampled listing points</div>
                                    <div class="lnote">Adds texture; can get visually busy.</div>
                                </div>
                            </label>
                            <label>
                                <input id="gemsToggle" type="checkbox" />
                                <div>
                                    <div class="lname">Low-Pressure Listings</div>
                                    <div class="lnote">Only appears when zoomed in.</div>
                                </div>
                            </label>
                        </div>
                    </div>

                    {borough_dropdown_html}

                    <div class="section" id="boroughLegendSection" style="display:none;">
                        <h3>Borough Pressure Legend</h3>
                        <div id="boroughLegendSlot" style="border:1px solid rgba(0,0,0,0.06);border-radius:12px;background:rgba(250,250,250,0.6);padding:10px;overflow:hidden;"></div>
                    </div>

                                        <div class="credit">
                                                <div><b>Developed by</b> <a href="https://laurenleek.eu" target="_blank" rel="noopener noreferrer" style="color:inherit;text-decoration:underline;">Lauren Leek</a></div>
                                                <div>London Airbnb diagnostics</div>
                                        </div>
        </div>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(sidebar))

    # --- Layers (professional + not overcrowded) ---
    footprints_fg = folium.FeatureGroup(name="Cluster footprints", show=True)
    centroids_fg = folium.FeatureGroup(name="Cluster centroids", show=True)
    sample_points_fg = folium.FeatureGroup(name="Sampled listing points", show=False)
    # Low-pressure listings as listing-level markers (toggleable)
    gems_layer = plugins.MarkerCluster(
        name="Low-Pressure Listings",
        show=False,
        disableClusteringAtZoom=15,
        spiderfyOnMaxZoom=True,
        removeOutsideVisibleBounds=True,
    )

    rng = np.random.default_rng(42)
    cluster_size_lookup = cluster_sizes.to_dict()

    clustered_df = lock.loc[lock["cluster_id"] != -1, ["cluster_id", "latitude", "longitude"]].copy()

    # --- Low-Pressure Listings (loose, structural, listing-level) ---
    # A "low-pressure listing" here is not a tourist hotspot; it's an individual listing with strong structure
    # (reviews/ratings relative to price + self-check-in + availability), plus separation from the biggest clusters.
    LON_CENTER = (51.509865, -0.118092)

    trap_ids = set(cluster_sizes.head(5).index.astype(int).tolist())

    # Precompute cluster centroids for distances
    centroids = (
        lock.loc[lock["cluster_id"] != -1]
        .groupby("cluster_id")[["latitude", "longitude"]]
        .mean()
        .rename(columns={"latitude": "c_lat", "longitude": "c_lon"})
    )

    # Distances: to center and to nearest trap centroid
    def _dist_to_center(row):
        return _haversine_km(float(row["c_lat"]), float(row["c_lon"]), LON_CENTER[0], LON_CENTER[1])

    centroids = centroids.copy()
    centroids["dist_center_km"] = centroids.apply(_dist_to_center, axis=1)

    trap_centroids = centroids.loc[centroids.index.astype(int).isin(trap_ids), ["c_lat", "c_lon"]]
    if len(trap_centroids) > 0:
        trap_points = trap_centroids.to_numpy()

        def _dist_to_nearest_trap(row):
            lat, lon = float(row["c_lat"]), float(row["c_lon"])
            return float(min(_haversine_km(lat, lon, float(t[0]), float(t[1])) for t in trap_points))

        centroids["dist_nearest_trap_km"] = centroids.apply(_dist_to_nearest_trap, axis=1)
    else:
        centroids["dist_nearest_trap_km"] = np.nan

    # Listing-level distances to nearest trap centroid (vectorized)
    if len(trap_centroids) > 0:
        trap_lat = trap_centroids["c_lat"].to_numpy(dtype=float)
        trap_lon = trap_centroids["c_lon"].to_numpy(dtype=float)
        lat = lock["latitude"].to_numpy(dtype=float)[:, None]
        lon = lock["longitude"].to_numpy(dtype=float)[:, None]

        # haversine distance to each trap centroid
        r = 6371.0
        phi1 = np.radians(lat)
        phi2 = np.radians(trap_lat[None, :])
        dphi = np.radians(trap_lat[None, :] - lat)
        dl = np.radians(trap_lon[None, :] - lon)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dl / 2) ** 2
        dist = 2 * r * np.arcsin(np.sqrt(a))
        dist_nearest_trap = dist.min(axis=1)
    else:
        dist_nearest_trap = np.zeros(len(lock), dtype=float)

    lock = lock.copy()
    lock["dist_nearest_trap_km"] = dist_nearest_trap

    # Listing-level gem score
    def _z_np(x: np.ndarray) -> np.ndarray:
        x = x.astype(float)
        m = np.nanmean(x)
        s = np.nanstd(x)
        return (x - m) / (s + 1e-9)

    rating_arr = lock["review_scores_rating"].to_numpy(dtype=float) if "review_scores_rating" in lock.columns else np.full(len(lock), np.nan)
    reviews_arr = lock["number_of_reviews"].to_numpy(dtype=float) if "number_of_reviews" in lock.columns else np.full(len(lock), np.nan)
    price_arr = lock["price_numeric"].to_numpy(dtype=float) if "price_numeric" in lock.columns else np.full(len(lock), np.nan)
    selfcheck_arr = lock["self_checkin_flag"].to_numpy(dtype=float) if "self_checkin_flag" in lock.columns else np.full(len(lock), np.nan)
    avail_arr = lock["availability_365"].to_numpy(dtype=float) if "availability_365" in lock.columns else np.full(len(lock), np.nan)
    dist_arr = lock["dist_nearest_trap_km"].to_numpy(dtype=float)

    # Loose scoring: reward structure, gently reward separation from traps
    gem_score = (
        0.9 * _z_np(np.nan_to_num(rating_arr, nan=np.nanmedian(rating_arr)))
        + 0.6 * _z_np(np.log1p(np.nan_to_num(reviews_arr, nan=np.nanmedian(reviews_arr))))
        + 0.3 * _z_np(np.nan_to_num(selfcheck_arr, nan=np.nanmedian(selfcheck_arr)))
        + 0.2 * _z_np(np.nan_to_num(avail_arr, nan=np.nanmedian(avail_arr)))
        - 0.7 * _z_np(np.log1p(np.nan_to_num(price_arr, nan=np.nanmedian(price_arr))))
        + 0.25 * np.tanh((dist_arr - 1.5) / 3.0)
    )

    lock["gem_score"] = gem_score

    # Candidate listings: not in trap clusters (but may be unclustered)
    not_trap = ~lock["cluster_id"].astype(int).isin(trap_ids)
    candidates = lock.loc[not_trap].copy()

    # Keep the map responsive: limit to top-N listings (and dedupe by listing id if available)
    candidates = candidates.sort_values("gem_score", ascending=False)
    if "id" in candidates.columns:
        candidates = candidates.drop_duplicates(subset=["id"], keep="first")
    candidates = candidates.head(MAX_LOW_PRESSURE_LISTINGS)

    print(f"Selecting top {len(candidates):,} low-pressure listings...")

    # Build footprints + centroids per cluster
    print("Rendering clusters (footprints, centroids, sampled points)...")
    for cid, group in clustered_df.groupby("cluster_id"):
        cid_int = int(cid)
        size = int(cluster_size_lookup.get(cid_int, len(group)))
        label = _cluster_label(size)

        # Compute centroid early (used for low-pressure listings too)
        lat = float(group["latitude"].mean())
        lon = float(group["longitude"].mean())

        # Convex hull footprint (requires >= 3 points)
        if len(group) >= 3:
            pts = group[["longitude", "latitude"]].to_numpy()
            # If points are collinear, ConvexHull can fail; skip footprint gracefully.
            try:
                hull = ConvexHull(pts)
                ring = pts[hull.vertices]
                latlon = [[float(y), float(x)] for x, y in ring]
                if latlon:
                    latlon.append(latlon[0])
                    folium.Polygon(
                        locations=latlon,
                        color=color_for(cid_int),
                        weight=2,
                        fill=True,
                        fill_color=color_for(cid_int),
                        fill_opacity=0.18,
                        tooltip=f"Cluster {cid_int} • {size:,} listings • {label}",
                    ).add_to(footprints_fg)
            except Exception:
                pass

        # Centroid bubble
        radius = float(np.clip(np.sqrt(size) * 0.35, 5.0, 18.0))
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color_for(cid_int),
            weight=2,
            fill=True,
            fill_color=color_for(cid_int),
            fill_opacity=0.85,
            tooltip=f"Cluster {cid_int} • {size:,} listings • {label}",
        ).add_to(centroids_fg)

        # No cluster-level gem markers; gems are listing-level and appear only when zoomed in + toggled.

        # Optional sampled points (adds texture without overload)
        sample_n = int(min(120, size))
        if sample_n > 0:
            sample_idx = rng.choice(len(group), size=min(sample_n, len(group)), replace=False)
            sample = group.iloc[sample_idx]
            for _, r in sample.iterrows():
                folium.CircleMarker(
                    location=[float(r["latitude"]), float(r["longitude"])],
                    radius=1.4,
                    color=color_for(cid_int),
                    weight=0,
                    fill=True,
                    fill_opacity=0.55,
                ).add_to(sample_points_fg)


    footprints_fg.add_to(m)
    centroids_fg.add_to(m)
    sample_points_fg.add_to(m)
    gems_layer.add_to(m)

    # Add gem listing markers (only for the dedicated gems layer)
    print("Rendering low-pressure listing markers...")
    for _, r in candidates.iterrows():
        lat = float(r["latitude"])
        lon = float(r["longitude"])
        name = html.escape(str(r.get("name", "")) or "")
        borough = html.escape(str(r.get(borough_col, "")) or "") if borough_col is not None else ""

        price = r.get("price_numeric", np.nan)
        rating = r.get("review_scores_rating", np.nan)
        reviews = r.get("number_of_reviews", np.nan)
        distt = r.get("dist_nearest_trap_km", np.nan)

        def _fmt(x, fmt):
            try:
                x = float(x)
            except Exception:
                return "—"
            return "—" if np.isnan(x) else fmt.format(x)

        popup = (
            "<div style='font-family:Segoe UI,Arial,sans-serif;'>"
            "<div style='font-weight:900;font-size:14px;margin-bottom:6px;'>Low-Pressure Listing</div>"
            f"<div style='font-size:12px;color:#111827;line-height:1.35;'><b>{name}</b></div>"
            f"<div style='font-size:12px;color:#6b7280;margin-top:4px;'>{borough}</div>"
            f"<div style='font-size:12px;color:#111827;margin-top:8px;'>"
            f"<b>Price</b>: {_fmt(price,'£{:.0f}')} &nbsp; "
            f"<b>Rating</b>: {_fmt(rating,'{:.2f}')} &nbsp; "
            f"<b>Reviews</b>: {_fmt(reviews,'{:.0f}')}"
            "</div>"
            f"<div style='font-size:12px;color:#111827;margin-top:4px;'><b>From tourist traps</b>: {_fmt(distt,'{:.1f} km')}</div>"
            "</div>"
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=3.0,
            color="#10b981",
            weight=1.5,
            fill=True,
            fill_color="#10b981",
            fill_opacity=0.70,
            tooltip="Low-Pressure Listing",
            popup=folium.Popup(popup, max_width=360),
        ).add_to(gems_layer)

    # No in-map label boxes: keep the map clean and let tooltips/side panel carry context.

    # --- Optional borough overlay (only if you provide a GeoJSON) ---
    borough_layer_name: str | None = None
    if borough_geojson is not None and housing is not None and borough_col is not None:
        def _norm_borough(s: str) -> str:
            s = str(s or "").strip().casefold()
            s = s.replace("&", "and")
            s = " ".join(s.split())
            return s

        try:
            housing_norm = housing.copy()

            # Support multiple schemas:
            # - xlsx: Borough,Housing (already normalized by _read_housing_stock_xlsx)
            # - legacy: borough,total_dwellings
            # - 2020 LA export: Code,Area,Number of properties 2020
            if "borough" in housing_norm.columns and "total_dwellings" in housing_norm.columns:
                housing_norm = housing_norm[["borough", "total_dwellings"]].copy()
                housing_norm["total_dwellings"] = pd.to_numeric(housing_norm["total_dwellings"], errors="coerce")
            elif "Borough" in housing_norm.columns and "Housing" in housing_norm.columns:
                housing_norm = housing_norm[["Borough", "Housing"]].rename(
                    columns={"Borough": "borough", "Housing": "total_dwellings"}
                )
                housing_norm["borough"] = housing_norm["borough"].astype(str).str.strip()
                housing_norm["total_dwellings"] = pd.to_numeric(housing_norm["total_dwellings"], errors="coerce")
            elif "Area" in housing_norm.columns and "Number of properties 2020" in housing_norm.columns:
                housing_norm = housing_norm[["Area", "Number of properties 2020"]].rename(
                    columns={"Area": "borough", "Number of properties 2020": "total_dwellings"}
                )
                housing_norm["total_dwellings"] = (
                    housing_norm["total_dwellings"]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .replace({"..": np.nan, "": np.nan, "nan": np.nan})
                )
                housing_norm["total_dwellings"] = pd.to_numeric(housing_norm["total_dwellings"], errors="coerce")
            else:
                raise ValueError("Unrecognized housing stock CSV schema")

            housing_norm["borough_norm"] = housing_norm["borough"].map(_norm_borough)

            counts = lock.groupby(borough_col).size().rename("airbnb_listings").reset_index().rename(columns={borough_col: "borough"})
            counts["borough_norm"] = counts["borough"].map(_norm_borough)

            merged = housing_norm.merge(
                counts[["borough_norm", "airbnb_listings"]],
                on="borough_norm",
                how="left",
            ).fillna({"airbnb_listings": 0})

            # Sanity-check housing stock: if it is (near-)perfectly proportional to listings,
            # the resulting % will be constant and misleading.
            ratios = np.where(
                merged["airbnb_listings"].astype(float) > 0,
                merged["total_dwellings"].astype(float) / merged["airbnb_listings"].astype(float),
                np.nan,
            )
            finite = np.isfinite(ratios)
            if finite.any():
                r_med = float(np.nanmedian(ratios[finite]))
                r_iqr = float(np.nanpercentile(ratios[finite], 75) - np.nanpercentile(ratios[finite], 25))
                # Flag suspicious when almost all boroughs share the same dwellings/listings ratio.
                if r_iqr < 1e-6 and 5.0 < r_med < 1000.0:
                    print(
                        "WARNING: Housing stock values look proportional to Airbnb listings (constant dwellings/listings ratio). "
                        "Borough pressure (% of housing stock) would be misleading. "
                        "Please replace the housing CSV with real dwelling counts to enable this layer."
                    )
                    raise ValueError("Housing stock appears proportional to listings")

            merged["airbnb_percent"] = np.where(
                merged["total_dwellings"].astype(float) > 0,
                merged["airbnb_listings"].astype(float) / merged["total_dwellings"].astype(float) * 100.0,
                np.nan,
            )

            by_norm = merged.set_index("borough_norm")[
                ["borough", "airbnb_listings", "total_dwellings", "airbnb_percent"]
            ].to_dict(orient="index")

            # Attach computed properties back onto each GeoJSON feature.
            pressure_vals: list[float] = []
            for feat in borough_geojson.get("features") or []:
                props = feat.setdefault("properties", {})
                geo_name = str(props.get("name") or "").strip()
                norm = _norm_borough(geo_name)
                rec = by_norm.get(norm)
                if rec is None:
                    props["borough"] = geo_name
                    props["airbnb_listings"] = 0
                    props["total_dwellings"] = None
                    props["airbnb_percent"] = None
                else:
                    props["borough"] = rec.get("borough") or geo_name
                    props["airbnb_listings"] = int(float(rec.get("airbnb_listings") or 0))
                    td = rec.get("total_dwellings")
                    props["total_dwellings"] = None if td is None or (isinstance(td, float) and np.isnan(td)) else float(td)
                    # Round for nicer tooltip display (keeps map computation stable)
                    ap = rec.get("airbnb_percent")
                    props["airbnb_percent"] = None if ap is None or (isinstance(ap, float) and np.isnan(ap)) else round(float(ap), 3)

                ap2 = props.get("airbnb_percent")
                if ap2 is not None:
                    try:
                        ap2f = float(ap2)
                    except Exception:
                        ap2f = np.nan
                    if np.isfinite(ap2f):
                        pressure_vals.append(ap2f)

            vals = np.asarray(pressure_vals, dtype=float)
            if len(vals) == 0:
                raise ValueError("No pressure values computed")

            # Discrete stepped shades across pressure levels (quantile bins).
            # Include 100th percentile to ensure the max is always covered.
            qs = np.nanpercentile(vals, [0, 20, 40, 60, 80, 95, 100])
            bins = np.unique(np.clip(qs, 0.0, None))
            if len(bins) < 2:
                bins = np.array([0.0, float(np.nanmax(vals) if np.isfinite(np.nanmax(vals)) else 1.0)])
            # Ensure the last edge is > first.
            if float(bins[-1]) <= float(bins[0]):
                bins = np.array([0.0, 1.0])

            # Use an ordered OrRd palette; pick as many steps as we have bin intervals.
            n_steps = int(len(bins) - 1)
            base_colors = list(bcm.linear.OrRd_09.colors)
            if n_steps <= len(base_colors):
                step_colors = base_colors[-n_steps:]
            else:
                # Fallback: sample along the linear map.
                lin = bcm.linear.OrRd_09.scale(float(bins[0]), float(bins[-1]))
                step_colors = [lin(float(bins[0]) + (i / max(n_steps - 1, 1)) * (float(bins[-1]) - float(bins[0]))) for i in range(n_steps)]

            pressure_cmap = bcm.StepColormap(
                colors=step_colors,
                index=bins.tolist(),
                vmin=float(bins[0]),
                vmax=float(bins[-1]),
                caption="Borough pressure (% of housing stock)",
            )

            def _borough_fill(p) -> str:
                try:
                    p = float(p)
                except Exception:
                    return "#bdbdbd"
                if not np.isfinite(p):
                    return "#bdbdbd"
                return pressure_cmap(p)

            borough_fg = folium.FeatureGroup(name="Borough pressure (normalized)", show=True)
            folium.GeoJson(
                data=borough_geojson,
                style_function=lambda feat: {
                    "fillColor": _borough_fill((feat.get("properties") or {}).get("airbnb_percent")),
                    "color": "#111827",
                    "weight": 1.0,
                    "fillOpacity": 0.35,
                },
                highlight_function=lambda feat: {"weight": 2.2, "color": "#000"},
                tooltip=folium.GeoJsonTooltip(
                    fields=["borough", "airbnb_percent", "airbnb_listings", "total_dwellings"],
                    aliases=["Borough", "Airbnb %", "Listings", "Housing stock"],
                    localize=True,
                    sticky=False,
                ),
            ).add_to(borough_fg)

            # Add the legend to the map.
            pressure_cmap.add_to(m)

            borough_fg.add_to(m)
            borough_layer_name = borough_fg.get_name()
        except Exception:
            # If anything goes wrong, we silently skip the borough layer.
            pass

    # One deferred UI script at the end: popup, borough dropdown, and left-panel layer toggles.
    # NOTE: This is injected into Folium's script block, so it must be raw JS
    # (no surrounding <script> tags), otherwise the HTML becomes invalid.
    ui_js = """
            (function(){
                var mapName = "__MAP_NAME__";
                var layers = {
                    footprints: "__FOOTPRINTS__",
                    centroids: "__CENTROIDS__",
                    borough: "__BOROUGH__",
                    sample: "__SAMPLE__",
                    gems: "__GEMS__"
                };
                var minGemsZoom = 13;

                var userWantsGems = false;
                var suppress = false;

                function getGlobal(name){
                    if (!name) return null;
                    try { return window[name]; } catch(e) { return null; }
                }
                function getMap(){ return getGlobal(mapName); }

                function safeAdd(map, layer){
                    suppress = true;
                    try { if (layer && !map.hasLayer(layer)) map.addLayer(layer); }
                    finally { suppress = false; }
                }
                function safeRemove(map, layer){
                    suppress = true;
                    try { if (layer && map.hasLayer(layer)) map.removeLayer(layer); }
                    finally { suppress = false; }
                }

                function setChecked(id, v){
                    var el = document.getElementById(id);
                    if (el) el.checked = !!v;
                }

                function init(){
                    if (window.__AIRBNB_DASH_INIT_DONE__) return true;
                    var map = getMap();
                    if (!map || typeof L === 'undefined') return false;

                    var footprints = getGlobal(layers.footprints);
                    var centroids = getGlobal(layers.centroids);
                    var borough = getGlobal(layers.borough);
                    var sample = getGlobal(layers.sample);
                    var gems = getGlobal(layers.gems);

                    // Critical: don't bind until all required layers exist.
                    // (Map is created before layers in Folium's JS output.)
                    if (!footprints || !centroids || !sample || !gems) return false;
                    if (layers.borough && layers.borough.length && !borough) return false;

                    function showHelp(){
                        var existing = document.getElementById('helpOverlay');
                        if (existing) {
                            existing.style.display = 'flex';
                            try { document.body.style.overflow = 'hidden'; } catch(e) {}
                            return;
                        }

                        var overlay = document.createElement('div');
                        overlay.id = 'helpOverlay';
                        overlay.style.position = 'fixed';
                        overlay.style.top = '0';
                        overlay.style.left = '0';
                        overlay.style.right = '0';
                        overlay.style.bottom = '0';
                        overlay.style.zIndex = '10000';
                        overlay.style.background = 'rgba(0,0,0,0.45)';
                        overlay.style.backdropFilter = 'blur(2px)';
                        overlay.style.display = 'flex';
                        overlay.style.alignItems = 'center';
                        overlay.style.justifyContent = 'center';
                        overlay.style.padding = '18px';
                        overlay.setAttribute('role', 'dialog');
                        overlay.setAttribute('aria-modal', 'true');

                        var card = document.createElement('div');
                        card.style.maxWidth = '560px';
                        card.style.width = '100%';
                        card.style.background = 'rgba(255,255,255,0.98)';
                        card.style.border = '1px solid rgba(0,0,0,0.10)';
                        card.style.borderRadius = '14px';
                        card.style.boxShadow = '0 14px 40px rgba(0,0,0,0.20)';
                        card.style.fontFamily = "Segoe UI,Arial,sans-serif";
                        card.style.color = '#111827';

                        var inner = document.createElement('div');
                                                inner.style.padding = '16px 18px 14px 18px';
                                                inner.innerHTML =
                                                        "<div style='display:flex;align-items:flex-start;justify-content:space-between;gap:12px;'>" +
                                                            "<div>" +
                                                                "<div style='font-weight:950;font-size:16px;line-height:1.15;'>How to read this map</div>" +
                                                                "<div style='margin-top:4px;font-size:12px;line-height:1.35;color:#6b7280;'>Clusters show where Airbnb concentrates; the borough overlay normalizes by housing stock, and low-pressure listings highlight strong individual listings when you zoom in.</div>" +
                                                            "</div>" +
                                                            "<button id='helpCloseX' aria-label='Close help' style='border:1px solid rgba(0,0,0,0.14);background:#fff;border-radius:10px;padding:6px 10px;font-weight:900;cursor:pointer;line-height:1;'>×</button>" +
                                                        "</div>" +

                                                        "<div style='margin-top:12px;padding:12px 12px;border:1px solid rgba(0,0,0,0.06);border-radius:12px;background:rgba(249,250,251,0.85);'>" +
                                                            "<div style='font-size:12px;color:#374151;line-height:1.55;'>" +
                                                                "<div style='font-weight:900;color:#111827;margin-bottom:8px;'>Features</div>" +
                                                                "<div style='display:grid;grid-template-columns: 1fr;gap:8px;'>" +
                                                                    "<div><b>Cluster footprints</b><div style='color:#6b7280;margin-top:2px;'>Simplified outlines of hotspots.</div></div>" +
                                                                    "<div><b>Centroids</b><div style='color:#6b7280;margin-top:2px;'>Bubbles sized by cluster count.</div></div>" +
                                                                    "<div><b>Borough pressure</b><div style='color:#6b7280;margin-top:2px;'>Listings as % of housing stock (if available).</div></div>" +
                                                                    "<div><b>Explore by borough</b><div style='color:#6b7280;margin-top:2px;'>Dropdown zooms to borough boundaries.</div></div>" +
                                                                    "<div><b>Low-Pressure Listings</b><div style='color:#6b7280;margin-top:2px;'>Toggle on + zoom in to show listing-level markers.</div></div>" +
                                                                "</div>" +
                                                            "</div>" +
                                                        "</div>" +

                                                        "<div style='margin-top:10px;display:flex;align-items:center;justify-content:space-between;gap:10px;'>" +
                                                            "<div style='font-size:12px;color:#6b7280;'>Tip: use the <b>Layers</b> section on the left. Press <b>Esc</b> to close this.</div>" +
                                                            "<button id='helpCloseBtn' style='border:1px solid rgba(0,0,0,0.14);background:#111827;color:#fff;border-radius:12px;padding:8px 12px;font-size:12px;font-weight:900;cursor:pointer;'>Got it</button>" +
                                                        "</div>";

                        card.appendChild(inner);
                        overlay.appendChild(card);
                        document.body.appendChild(overlay);

                        try { document.body.style.overflow = 'hidden'; } catch(e) {}

                        function close(){
                            overlay.style.display = 'none';
                            try { document.body.style.overflow = ''; } catch(e) {}
                        }
                        overlay.addEventListener('click', function(e){ if (e.target === overlay) close(); });
                        document.getElementById('helpCloseBtn').addEventListener('click', close);
                        document.getElementById('helpCloseX').addEventListener('click', close);

                        document.addEventListener('keydown', function(e){
                            if (overlay.style.display === 'flex' && (e.key === 'Escape' || e.key === 'Esc')) close();
                        });
                    }

                    // Show dismissable help on open
                    try { showHelp(); } catch(e) {}

                    // Borough dropdown
                    try {
                        var sel = document.getElementById('boroughSelect');
                        if (sel && window.__BOROUGH_BOUNDS__) {
                            sel.addEventListener('change', function(){
                                var b = window.__BOROUGH_BOUNDS__[sel.value];
                                if (b) map.fitBounds(b, {padding:[20,20]});
                            });
                        }
                    } catch(e) {}

                    // If borough overlay failed to load, disable the toggle.
                    var boroughCb = document.getElementById('toggleBorough');
                    if (boroughCb && !borough) boroughCb.disabled = true;

                    // Move borough pressure legend under the left panel (if present)
                    try {
                        var slot = document.getElementById('boroughLegendSlot');
                        var sec = document.getElementById('boroughLegendSection');
                        var legend = document.querySelector('.legend.leaflet-control');
                        if (slot && legend) {
                            slot.appendChild(legend);
                            legend.style.position = 'static';
                            legend.style.margin = '0';
                            legend.style.background = 'transparent';
                            legend.style.border = '0';
                            legend.style.boxShadow = 'none';
                            if (sec) sec.style.display = 'block';
                        }
                    } catch(e) {}

                    // Fullscreen: Leaflet's plugin can change stacking contexts.
                    // Ensure the left dashboard stays visible.
                    function ensurePanelVisible(){
                        try {
                            var panel = document.getElementById('panel');
                            if (!panel) return;
                            // Keep it out of any element-fullscreen stacking issues.
                            if (panel.parentElement !== document.body) {
                                document.body.appendChild(panel);
                            }
                            panel.style.display = 'block';
                            panel.style.visibility = 'visible';
                            panel.style.opacity = '1';
                            panel.style.position = 'fixed';
                            panel.style.left = '24px';
                            panel.style.top = '24px';
                            panel.style.zIndex = '2147483647';
                        } catch(e) {}
                    }

                    try {
                        // Run once in case the page loads already in fullscreen.
                        ensurePanelVisible();
                        document.addEventListener('fullscreenchange', ensurePanelVisible);
                        document.addEventListener('webkitfullscreenchange', ensurePanelVisible);
                        document.addEventListener('mozfullscreenchange', ensurePanelVisible);
                        document.addEventListener('MSFullscreenChange', ensurePanelVisible);
                    } catch(e) {}

                    function enforceGems(){
                        var z = map.getZoom();
                        if (userWantsGems && z >= minGemsZoom) safeAdd(map, gems);
                        else safeRemove(map, gems);
                    }

                    function toggleLayer(id, layer){
                        var cb = document.getElementById(id);
                        if (!cb) return;
                        cb.addEventListener('change', function(){
                            if (id === 'gemsToggle') {
                                userWantsGems = !!cb.checked;
                                enforceGems();
                                return;
                            }
                            if (cb.checked) safeAdd(map, layer);
                            else safeRemove(map, layer);
                        });
                    }

                    // Hook up sidebar toggles
                    toggleLayer('toggleFootprints', footprints);
                    toggleLayer('toggleCentroids', centroids);
                    toggleLayer('toggleBorough', borough);
                    toggleLayer('toggleSample', sample);
                    toggleLayer('gemsToggle', gems);

                    // Help button
                    try {
                        var hb = document.getElementById('helpBtn');
                        if (hb) hb.addEventListener('click', function(e){ e.preventDefault(); showHelp(); });
                    } catch(e) {}

                    // Keep gems zoom-gated
                    map.on('zoomend', enforceGems);

                    // Initialize checkbox states to match current layers
                    setChecked('toggleFootprints', !!(footprints && map.hasLayer(footprints)));
                    setChecked('toggleCentroids', !!(centroids && map.hasLayer(centroids)));
                    setChecked('toggleBorough', !!(borough && map.hasLayer(borough)));
                    setChecked('toggleSample', !!(sample && map.hasLayer(sample)));
                    setChecked('gemsToggle', false);
                    userWantsGems = false;
                    enforceGems();
                    window.__AIRBNB_DASH_INIT_DONE__ = true;
                    return true;
                }

                (function wait(n){
                    if (init()) return;
                    if (n <= 0) return;
                    setTimeout(function(){ wait(n - 1); }, 60);
                })(250);
            })();
    """
    ui_js = (
        ui_js.replace("__MAP_NAME__", m.get_name())
        .replace("__FOOTPRINTS__", footprints_fg.get_name())
        .replace("__CENTROIDS__", centroids_fg.get_name())
        .replace("__SAMPLE__", sample_points_fg.get_name())
        .replace("__GEMS__", gems_layer.get_name())
        .replace("__BOROUGH__", borough_layer_name or "")
    )
    # Important: inject this into the script section so it runs after Folium has
    # defined the Leaflet map + layer variables.
    m.get_root().script.add_child(folium.Element(ui_js))

    out_path = output_html or OUTPUT_HTML
    print(f"Saving HTML: {out_path}")
    m.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the Airbnb adjusted vacancy Folium map")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output HTML path (default: {OUTPUT_HTML})",
    )
    args = parser.parse_args()

    main(output_html=Path(args.output) if args.output else None)

