#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def season_label(dt: pd.Timestamp) -> str:
    """
    Season runs 1 Oct → 30 Sep next year.
    Example: 2023-10-01 → '2023-2024', 2024-06-30 → '2023-2024'
    """
    y = dt.year
    if dt.month >= 10:
        return f"{y}-{y+1}"
    else:
        return f"{y-1}-{y}"

def main(input_path: str, outdir: str, date_col: str = "DateTime", site_col: str = "SITE_NAME"):
    inp = Path(input_path)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Load
    df = pd.read_csv(inp)
    # Be robust to day-first dates and mixed formats
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    # Drop rows without valid timestamps or site names
    df = df.dropna(subset=[date_col, site_col]).copy()

    # De-duplicate conservative key (site + timestamp)
    df = df.sort_values([site_col, date_col]).drop_duplicates(
        subset=[site_col, date_col], keep="first"
    )

    # --- Season column (Oct→Sep)
    df["Season"] = df[date_col].apply(season_label)

    # --- Metrics
    # 1) Total counts by Season
    counts_by_season = (
        df.groupby("Season", as_index=False)
          .size()
          .rename(columns={"size": "n_rows"})
          .sort_values("Season")
    )

    # 2) Counts by Site × Season
    counts_site_season = (
        df.groupby([site_col, "Season"], as_index=False)
          .size()
          .rename(columns={"size": "n_rows"})
          .sort_values([site_col, "Season"])
    )

    # 3) (Optional) Pivot table: rows=Site, cols=Season, values=n_rows
    pivot_site_season = counts_site_season.pivot(
        index=site_col, columns="Season", values="n_rows"
    ).fillna(0).astype(int).sort_index()

    # --- Save
    counts_by_season.to_csv(out / "counts_by_season.csv", index=False)
    counts_site_season.to_csv(out / "counts_by_site_season_long.csv", index=False)
    pivot_site_season.to_csv(out / "counts_by_site_season_wide.csv")

    # --- Quick console summary
    print(f"\nInput: {inp}")
    print(f"Rows after cleaning: {len(df):,}")
    print(f"Date range: {df[date_col].min()} → {df[date_col].max()}")
    print("\nCounts by season (head):")
    print(counts_by_season.head(10).to_string(index=False))
    print("\nTop 10 site×season cells by count:")
    print(counts_site_season.sort_values('n_rows', ascending=False)
                         .head(10).to_string(index=False))
    print(f"\nWrote:\n  - {out/'counts_by_season.csv'}"
          f"\n  - {out/'counts_by_site_season_long.csv'}"
          f"\n  - {out/'counts_by_site_season_wide.csv'}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Season & site counts (Oct→Sep season).")
    p.add_argument("--input", required=True, help="Path to cleaned CSV (with DateTime & SITE_NAME).")
    p.add_argument("--outdir", default="reports/season_counts", help="Output directory for CSVs.")
    p.add_argument("--date-col", default="DateTime", help="Date column name.")
    p.add_argument("--site-col", default="SITE_NAME", help="Site column name.")
    args = p.parse_args()
    main(args.input, args.outdir, date_col=args.date_col, site_col=args.site_col)
