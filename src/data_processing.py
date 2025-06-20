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
    silent: bool = False,
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
        if dups and not silent:
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
    silent: bool = False,
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
    if num_gaps and not silent:
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

# ----------------------------------------------------------------------
# 4. Next-day prediction feature engineering
# ----------------------------------------------------------------------
def create_nextday_features(df: pd.DataFrame, target_column: str, window_hours: int = 24, silent: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Creates features for next-day prediction using sliding windows.
    
    Args:
        df: DataFrame with datetime index and features
        target_column: Name of the target column (e.g., 'Gross Power (MW)')
        window_hours: Number of hours to use as input window (default: 24)
    
    Returns:
        Tuple of (X_features, y_target) where:
        - X_features: DataFrame with engineered features for each day
        - y_target: Series with next day's target values
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index")
    
    # Ensure hourly frequency
    if df.index.freq != 'H' and pd.infer_freq(df.index) != 'H':
        if not silent:
            logger.warning("Data is not hourly. Resampling to hourly frequency...")
        df = df.resample('H').mean().interpolate()
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_column]
    if not feature_cols:
        raise ValueError("No feature columns found")
    
    X_list = []
    y_list = []
    timestamps = []
    
    # Create sliding windows
    for i in range(window_hours, len(df) - 24):  # -24 to ensure we have next day's target
        # Get current window (e.g., 24 hours of features)
        window_data = df.iloc[i-window_hours:i][feature_cols]
        
        # Flatten the window into a single row
        window_features = window_data.values.flatten()
        
        # Create feature names for the flattened window
        feature_names = []
        for hour in range(window_hours):
            for feature in feature_cols:
                feature_names.append(f"{feature}_hour_{hour}")
        
        # Add daily aggregates as additional features
        daily_aggregates = []
        aggregate_names = []
        
        for feature in feature_cols:
            daily_mean = window_data[feature].mean()
            daily_min = window_data[feature].min()
            daily_max = window_data[feature].max()
            daily_last = window_data[feature].iloc[-1]
            daily_std = window_data[feature].std()
            
            daily_aggregates.extend([daily_mean, daily_min, daily_max, daily_last, daily_std])
            aggregate_names.extend([
                f"{feature}_daily_mean",
                f"{feature}_daily_min", 
                f"{feature}_daily_max",
                f"{feature}_daily_last",
                f"{feature}_daily_std"
            ])
        
        # Combine window features and aggregates
        all_features = np.concatenate([window_features, daily_aggregates])
        all_feature_names = feature_names + aggregate_names
        
        # Get next day's target (24 hours)
        next_day_target = df.iloc[i:i+24][target_column].values
        
        if len(next_day_target) == 24:
            X_list.append(all_features)
            y_list.append(next_day_target)
            # Store the timestamp for the beginning of the input window
            timestamps.append(df.index[i - window_hours])

    if not X_list:
        logger.warning("No samples were created. The dataset might be too short for the given window size.")
        # Return empty DataFrames with correct structure
        return pd.DataFrame(columns=all_feature_names), pd.DataFrame()

    # Create DataFrames
    X_df = pd.DataFrame(X_list, columns=all_feature_names)
    y_df = pd.DataFrame(y_list)
    
    # Set the timestamp as the index for X, so it's preserved
    X_df.index = pd.to_datetime(timestamps)
    X_df.index.name = 'timestamp'

    logger.info(f"Created {len(X_df)} samples with {len(all_feature_names)} features")
    logger.info(f"Target shape: {y_df.shape}")

    return X_df, y_df

def prepare_nextday_input(today_data: pd.DataFrame, target_column: str, window_hours: int = 24) -> pd.DataFrame:
    """
    Prepares input features for next-day prediction from today's data.
    
    Args:
        today_data: DataFrame with today's hourly data (should have 24 rows)
        target_column: Name of the target column
        window_hours: Number of hours to use as input window
    
    Returns:
        DataFrame with engineered features ready for prediction
    """
    if len(today_data) != window_hours:
        raise ValueError(f"Today's data should have exactly {window_hours} rows (hours)")
    
    # Separate features and target
    feature_cols = [col for col in today_data.columns if col != target_column]
    if not feature_cols:
        raise ValueError("No feature columns found")
    
    # Get the window data (last window_hours)
    window_data = today_data[feature_cols]
    
    # Flatten the window into a single row
    window_features = window_data.values.flatten()
    
    # Create feature names for the flattened window
    feature_names = []
    for hour in range(window_hours):
        for feature in feature_cols:
            feature_names.append(f"{feature}_hour_{hour}")
    
    # Add daily aggregates as additional features
    daily_aggregates = []
    aggregate_names = []
    
    for feature in feature_cols:
        daily_mean = window_data[feature].mean()
        daily_min = window_data[feature].min()
        daily_max = window_data[feature].max()
        daily_last = window_data[feature].iloc[-1]
        daily_std = window_data[feature].std()
        
        daily_aggregates.extend([daily_mean, daily_min, daily_max, daily_last, daily_std])
        aggregate_names.extend([
            f"{feature}_daily_mean",
            f"{feature}_daily_min", 
            f"{feature}_daily_max",
            f"{feature}_daily_last",
            f"{feature}_daily_std"
        ])
    
    # Combine window features and aggregates
    all_features = np.concatenate([window_features, daily_aggregates])
    all_feature_names = feature_names + aggregate_names
    
    # Create DataFrame
    input_df = pd.DataFrame([all_features], columns=all_feature_names)
    
    logger.info(f"Prepared input with {len(input_df.columns)} features")
    
    return input_df 