#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

EXCEED = 280

def season_label(dt: pd.Timestamp) -> str:
    """Season runs 1 Oct → 30 Sep next year."""
    y = dt.year
    return f"{y}-{y+1}" if dt.month >= 10 else f"{y-1}-{y}"

def main(input_path: str, out_csv: str):
    inp = Path(input_path)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    # Processed CSV uses day-first format like '13/11/2013 11:00'
    df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["DateTime", "SITE_NAME", "Enterococci"]).copy()

    # Build season + exceed flag
    df["Season"] = df["DateTime"].apply(season_label)
    df["Exceed"] = df["Enterococci"] > EXCEED

    # Count exceedances per site-season
    counts = (
        df.groupby(["SITE_NAME", "Season"], as_index=False)["Exceed"]
          .sum()
          .rename(columns={"Exceed": "n_exceedances"})
          .sort_values(["SITE_NAME", "Season"])
    )

    counts.to_csv(out, index=False)

    print(f"Input rows: {len(df):,}")
    print(f"Date range: {df['DateTime'].min()} → {df['DateTime'].max()}")
    print(counts.head(12).to_string(index=False))
    print(f"\nWrote: {out.resolve()}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Count Enterococci exceedances (>280) per site-season.")
    p.add_argument("--input", required=True, help="Path to cleaned CSV (must include DateTime, SITE_NAME, Enterococci).")
    p.add_argument("--out", default="reports/exceedance_counts_per_site_season.csv",
                   help="Output CSV path.")
    args = p.parse_args()
    main(args.input, args.out)
