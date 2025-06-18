"""
data_processing.py
--------------------
Functions for loading, cleaning, and preparing time-series data for modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, IO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 1. I/O helpers
# ----------------------------------------------------------------------
def load_and_parse(
    file_source: Union[str, Path, IO[bytes]],
    datetime_col: str = None,
    start_date: str = None,
    freq: str = None,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    - Reads CSV/Parquet/Feather/Excel into pandas from a path or file-like object.
    - Parses the datetime column and sets it as a timezone-naive index.
    - If no datetime column is provided, generates one from a start date and frequency.
    - Sorts chronologically.
    """
    file_name = ""
    if hasattr(file_source, 'name'):
        file_name = file_source.name.lower()
    elif isinstance(file_source, (str, Path)):
        file_name = str(file_source).lower()

    try:
        if file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_source)
        elif file_name.endswith((".parquet", ".pq")):
            df = pd.read_parquet(file_source)
        elif file_name.endswith((".feather", ".ft")):
            df = pd.read_feather(file_source)
        else: # Default to CSV
            df = pd.read_csv(file_source)
    except Exception as e:
        raise ValueError(f"Error reading file {file_name}: {e}")

    if datetime_col:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
        if df[datetime_col].isna().any():
            raise ValueError("Some rows have unparsable timestamps. Fix these first!")
        df = df.sort_values(datetime_col).set_index(datetime_col)
    elif start_date and freq:
        index = pd.date_range(start=start_date, periods=len(df), freq=freq, name="datetime")
        df.set_index(index, inplace=True)
        df.sort_index(inplace=True)
    else:
        raise ValueError("Either 'datetime_col' or both 'start_date' and 'freq' must be provided.")

    if drop_duplicates:
        before = len(df)
        df = df[~df.index.duplicated(keep="first")]
        dups = before - len(df)
        if dups:
            logger.info(f"Dropped {dups} duplicate rows")

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
        logger.info(f"Inserted {num_gaps} gap rows. Interpolating...")

    df_uniform.interpolate(
        method=interp_method, limit=limit, limit_direction="both", inplace=True
    )

    obj_cols = df_uniform.select_dtypes(include="object").columns
    df_uniform[obj_cols] = df_uniform[obj_cols].ffill()

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
        logger.info(f"Inferred constant frequency: {inferred}") 