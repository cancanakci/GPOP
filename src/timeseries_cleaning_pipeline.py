"""
timeseries_cleaning_pipeline.py
--------------------------------
Reusable utility functions to load, inspect, and clean geothermal plant
time‑series data so it's ready for modelling with XGBoost and for
meaningful feature extrapolation.

Usage example
-------------
from timeseries_cleaning_pipeline import (
    load_and_parse,
    enforce_frequency,
    sanity_checks,
    detect_and_impute_outliers,
    add_rolling_features,
    train_test_split_time
)

df_raw = load_and_parse("raw_data.csv",
                        datetime_col="timestamp",
                        local_tz="Europe/Istanbul")

df_clean = enforce_frequency(df_raw, freq="H")  # hourly resolution
sanity_checks(df_clean)

numeric_cols = [c for c in df_clean.columns if df_clean[c].dtype != "object"]
df_clean = detect_and_impute_outliers(df_clean, numeric_cols)

df_feat = add_rolling_features(df_clean,
                               cols=["gross_power", "brine_rate"],
                               windows=[3, 6, 24])

train, test = train_test_split_time(df_feat, test_size=0.2)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union


# ----------------------------------------------------------------------
# 1. I/O helpers
# ----------------------------------------------------------------------
def load_and_parse(
    file_path: Union[str, Path],
    datetime_col: str,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    - Reads CSV/Parquet/Feather into pandas.
    - Parses the datetime column and sets it as index.
    - Sorts chronologically.
    """
    path = Path(file_path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".feather", ".ft"}:
        df = pd.read_feather(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    if df[datetime_col].isna().any():
        raise ValueError("Some rows have unparsable timestamps. Fix these first!")

    df = (
        df.sort_values(datetime_col)
        .set_index(datetime_col)
    )

    if drop_duplicates:
        before = len(df)
        df = df[~df.index.duplicated(keep="first")]
        dups = before - len(df)
        if dups:
            print(f"[load_and_parse] Dropped {dups} duplicate rows")

    return df


# ----------------------------------------------------------------------
# 2. Uniform frequency enforcement
# ----------------------------------------------------------------------
def enforce_frequency(
    df: pd.DataFrame,
    freq: str,
    interp_method: str = "linear",
    limit: int = None,
) -> pd.DataFrame:
    """
    Ensures the DataFrame index has a uniform DateTimeIndex at the given `freq`.
    Missing rows are inserted and interpolated.

    Parameters
    ----------
    freq : str
        Pandas frequency alias, e.g. "H" for hourly, "15T" for 15‑minute.
    interp_method : str
        One of pandas interpolate methods ("linear", "time", "akima", etc.).
    limit : int
        Max number of consecutive NaNs to fill. `None` = unlimited.

    Returns
    -------
    DataFrame with uniform frequency and no missing values.
    """
    full_index = pd.date_range(
        df.index.min(),
        df.index.max(),
        freq=freq,
        name=df.index.name,
    )
    df_uniform = df.reindex(full_index)

    num_gaps = df_uniform.isna().any(axis=1).sum()
    if num_gaps:
        print(f"[enforce_frequency] Inserted {num_gaps} gap rows. Interpolating...")

    df_uniform.interpolate(
        method=interp_method, limit=limit, limit_direction="both", inplace=True
    )

    # After interpolation, propagate object columns forward
    obj_cols = df_uniform.select_dtypes(include="object").columns
    df_uniform[obj_cols] = df_uniform[obj_cols].fillna(method="ffill")

    if df_uniform.isna().any().any():
        raise RuntimeError(
            "There are still NaNs after interpolation. "
            "Consider a different strategy or inspect the data."
        )

    return df_uniform


# ----------------------------------------------------------------------
# 3. Sanity checks
# ----------------------------------------------------------------------
def sanity_checks(df: pd.DataFrame) -> None:
    """
    Quick uniformity checks:
    - Index uniqueness
    - Monotonicity
    - Continuous coverage at constant frequency
    """
    idx = df.index
    if not idx.is_monotonic_increasing:
        raise ValueError("DatetimeIndex is not monotonic increasing")

    if idx.has_duplicates:
        raise ValueError("DatetimeIndex contains duplicates")

    inferred = pd.infer_freq(idx)
    if inferred is None:
        raise ValueError("Could not infer a constant frequency in DatetimeIndex")
    else:
        print(f"[sanity_checks] Inferred constant frequency: {inferred}")


# ----------------------------------------------------------------------
# 4. Outlier detection & imputation
# ----------------------------------------------------------------------
def detect_and_impute_outliers(
    df: pd.DataFrame, cols: List[str], z_threshold: float = 6.0
) -> pd.DataFrame:
    """
    Detects univariate outliers above `z_threshold` standard deviations and treats
    them as missing before calling interpolate.

    Returns
    -------
    DataFrame with outliers replaced by interpolated values.
    """
    df_out = df.copy()
    for col in cols:
        series = df_out[col]
        z_scores = (series - series.mean()) / series.std()
        mask = z_scores.abs() > z_threshold
        if mask.any():
            print(
                f"[detect_and_impute_outliers] {mask.sum()} outliers found in '{col}'."
            )
            df_out.loc[mask, col] = np.nan

    df_out.interpolate(method="linear", inplace=True, limit_direction="both")
    return df_out


# ----------------------------------------------------------------------
# 5. Feature engineering helpers
# ----------------------------------------------------------------------
def add_rolling_features(
    df: pd.DataFrame,
    cols: List[str],
    windows: List[int],
    min_periods: int = 1,
    center: bool = False,
) -> pd.DataFrame:
    """
    Adds simple rolling‑mean features for each `cols` × `windows` combination.
    """
    df_roll = df.copy()
    for w in windows:
        rolled = df[cols].rolling(window=w, min_periods=min_periods, center=center)
        df_roll = df_roll.join(
            rolled.mean().add_suffix(f"_ma{w}"),
            how="left",
        )
    return df_roll


# ----------------------------------------------------------------------
# 6. Train/test split for time‑series
# ----------------------------------------------------------------------
def train_test_split_time(
    df: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple chronological split that keeps the most recent part as test set.
    """
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut], df.iloc[cut:]


# ----------------------------------------------------------------------
# 7. Quick CLI driver
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean geothermal time‑series data for XGBoost modelling."
    )
    parser.add_argument("infile", type=str, help="Path to raw CSV file")
    parser.add_argument(
        "--datetime-col",
        type=str,
        default="timestamp",
        help="Name of the datetime column",
    )
    parser.add_argument(
        "--freq", type=str, default="H", help="Target frequency (e.g. 'H', '15T')"
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="clean_data.parquet",
        help="Output file (Parquet recommended)",
    )
    args = parser.parse_args()

    df0 = load_and_parse(args.infile, args.datetime_col)
    df1 = enforce_frequency(df0, freq=args.freq)
    sanity_checks(df1)

    num_cols = df1.select_dtypes(include="number").columns.tolist()
    df2 = detect_and_impute_outliers(df1, num_cols)

    df2.to_parquet(args.outfile)
    print(f"✅ Cleaned data saved to {args.outfile}") 