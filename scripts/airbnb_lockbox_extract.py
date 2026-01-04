#!/usr/bin/env python3
"""
airbnb_lockbox_extract.py

Extracts Airbnb self check-in (lockbox / smartlock) listings for London
from a single InsideAirbnb listings.csv file.

Usage:
    python airbnb_lockbox_extract.py \
        --input listings.csv \
        --output-prefix london_airbnb

Optional:
    python airbnb_lockbox_extract.py \
        --url "http://data.insideairbnb.com/united-kingdom/england/london/2025-01-01/visualisations/listings.csv" \
        --output-prefix london_airbnb_2025_01_01
"""

import argparse
import io
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"


# ------------------------
# Helpers
# ------------------------

def download_csv(url: str) -> pd.DataFrame:
    """Download a CSV from URL into a pandas DataFrame."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def load_input(input_path: Optional[str], url: Optional[str]) -> pd.DataFrame:
    """Load listings data from a local file or URL."""
    if input_path and url:
        raise ValueError("Please provide EITHER --input OR --url, not both.")
    if not input_path and not url:
        raise ValueError("You must provide either --input (path) or --url.")

    if input_path:
        path = Path(input_path)
        if not path.exists():
            # Try a gzip variant (e.g., listings.csv -> listings.csv.gz)
            gz = Path(str(path) + ".gz")
            if gz.exists():
                path = gz
            else:
                raise FileNotFoundError(f"Input file not found: {path}")
        print(f"[INFO] Loading listings from {path}")
        df = pd.read_csv(path, compression='infer')
    else:
        print(f"[INFO] Downloading listings from {url}")
        df = download_csv(url)

    print(f"[INFO] Loaded {len(df):,} rows, {len(df.columns)} columns.")
    return df


def infer_self_checkin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer a boolean self_checkin flag using typical InsideAirbnb fields.

    Tries in this order:
    1. 'self_checkin' column if present (already boolean or Y/N).
    2. 'has_availability' / 'host_verifications' string containing 'self check-in' or similar.
    3. Fallback: create empty False column and warn.
    """

    df = df.copy()

    # 1) Direct self_checkin column (newer dumps sometimes have this)
    if "self_checkin" in df.columns:
        print("[INFO] Using existing 'self_checkin' column.")
        col = df["self_checkin"]

        # Normalize to boolean
        if col.dtype == bool:
            df["self_checkin_flag"] = col
        else:
            df["self_checkin_flag"] = (
                col.astype(str)
                   .str.strip()
                   .str.lower()
                   .isin(["1", "true", "t", "yes", "y"]) 
            )
        return df

    # 2) Try to infer from 'host_verifications' or 'amenities'
    possible_cols = [c for c in df.columns if "verification" in c.lower()] + \
                    [c for c in df.columns if "amenit" in c.lower()]

    for colname in possible_cols:
        col = df[colname].astype(str).str.lower()
        if col.str.contains("self check", na=False).any() or col.str.contains("self-check", na=False).any():
            print(f"[INFO] Inferring self_checkin_flag from column '{colname}'.")
            df["self_checkin_flag"] = col.str.contains("self check", na=False) | \
                                      col.str.contains("self-check", na=False)
            return df

    # 3) Nothing found: fallback
    print("[WARN] No direct self-checkin info found. Creating 'self_checkin_flag' as False.")
    df["self_checkin_flag"] = False
    return df


def select_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns we care about for the lockbox / self-check-in project.
    Adjust this as needed.
    """
    wanted = [
        "id",
        "listing_url",
        "name",
        "room_type",
        "property_type",
        "neighbourhood_cleansed",
        "latitude",
        "longitude",
        "price",
        "availability_365",
        "minimum_nights",
        "maximum_nights",
        "calculated_host_listings_count",
        "number_of_reviews",
        "review_scores_rating",
        "self_checkin_flag",
    ]

    # Keep those that exist
    existing = [c for c in wanted if c in df.columns]
    missing = [c for c in wanted if c not in df.columns]

    if missing:
        print(f"[INFO] Missing columns (not fatal): {missing}")

    out = df[existing].copy()
    print(f"[INFO] Reduced to {len(existing)} columns.")
    return out


def clean_price_column(df: pd.DataFrame) -> pd.DataFrame:
    """Convert price strings like '$120.00' to numeric."""
    if "price" not in df.columns:
        return df

    df = df.copy()
    df["price_raw"] = df["price"]

    def _parse_price(x):
        if pd.isna(x):
            return None
        s = str(x)
        # Strip currency symbols and commas
        s = s.replace("$", "").replace("£", "").replace("€", "").replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None

    df["price"] = df["price"].apply(_parse_price)
    return df


def main():
    parser = argparse.ArgumentParser(description="Extract Airbnb lockbox/self-check-in listings for London.")
    parser.add_argument("--input", type=str, help="Path to InsideAirbnb listings.csv")
    parser.add_argument("--url", type=str, help="URL to InsideAirbnb listings.csv (if not using --input)")
    parser.add_argument("--output-prefix", type=str, default=str(OUTPUTS_DIR / "london_airbnb"),
                        help="Prefix for output files (default: london_airbnb)")

    # Use parse_known_args so extra args injected by environments (Jupyter kernels)
    # don't cause the script to exit with an error.
    args, unknown = parser.parse_known_args()

    # If neither --input nor --url provided, try a local 'listings.csv(.gz)' fallback.
    if not args.input and not args.url:
        possible = [
            DATA_DIR / "listings.csv.gz",
            DATA_DIR / "listings.csv",
            ROOT_DIR / "listings.csv.gz",
            ROOT_DIR / "listings.csv",
        ]
        for p in possible:
            if p.exists():
                args.input = str(p)
                print(f"[INFO] No --input/--url provided; using local file {p}")
                break

    try:
        df_raw = load_input(args.input, args.url)
    except Exception as e:
        print(f"[ERROR] Failed to load input: {e}", file=sys.stderr)
        # In interactive environments (notebooks) avoid SystemExit which is noisy.
        if 'get_ipython' in globals():
            return
        sys.exit(1)

    # Infer self-check-in
    df = infer_self_checkin(df_raw)

    # Optional: filter to London if multiple cities sneak in
    # (often not needed if you download only the London file)
    if "city" in df.columns:
        before = len(df)
        df = df[df["city"].astype(str).str.contains("london", case=False, na=False)]
        print(f"[INFO] Filtered to London: {len(df):,} rows (from {before:,}).")

    # Clean + select columns
    df = clean_price_column(df)
    df = select_relevant_columns(df)

    # Quick info
    total = len(df)
    self_checkin_count = df["self_checkin_flag"].sum()
    print(f"[INFO] Total listings kept: {total:,}")
    print(f"[INFO] Self check-in / lockbox listings: {self_checkin_count:,} "
          f"({self_checkin_count / max(total,1):.1%})")

    # Output
    out_prefix = Path(args.output_prefix)
    csv_path = out_prefix.with_suffix(".csv")
    parquet_path = out_prefix.with_suffix(".parquet")

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    print(f"[INFO] Saved CSV to:     {csv_path}")
    print(f"[INFO] Saved Parquet to: {parquet_path}")
    print("[DONE] You can now use these for clustering / mapping.")


if __name__ == "__main__":
    main()
