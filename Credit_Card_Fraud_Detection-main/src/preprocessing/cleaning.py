# cleaning utilities: col drops, type casting, basic validation

import pandas as pd
from src import config

def cast_datetimes(df):
    df[config.DATETIME_COL] = pd.to_datetime(df[config.DATETIME_COL])
    df[config.DOB_COL]      = pd.to_datetime(df[config.DOB_COL])

    return df

# remove cols that carry no signal (IDs)
def drop_initial_cols(df):
    cols_to_drop = [c for c in config.DROP_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)

# remove cols that have been replaced by engineered features
def drop_engineered_sources(df):
    cols_to_drop = [c for c in config.DROP_AFTER_ENGINEERING if c in df.columns]
    return df.drop(columns=cols_to_drop)

# drop raw lat/lon cols once distance_km has been computed
# controlled by config.DROP_COORDS_AFTER_DISTANCE
def drop_coords_if_configured(df):
    if not config.DROP_COORDS_AFTER_DISTANCE:
        return df
 
    coord_cols = [
        config.CUSTOMER_LAT, config.CUSTOMER_LON,
        config.MERCHANT_LAT, config.MERCHANT_LON,
    ]
    cols_to_drop = [c for c in coord_cols if c in df.columns]
    return df.drop(columns=cols_to_drop)

def validate(df):

    # check for nulls
    null_counts = df.isnull().sum()
    if null_counts.any():
        bad_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"null values found after preprocessing: {bad_cols}")
    
    # make sure target col present
    if config.TARGET_COL not in df.columns:
        raise ValueError(f"target column {config.TARGET_COL} is missing from dataframe")
    
    # age should be positive
    if "age" in df.columns and (df["age"] <= 0).any():
        raise ValueError("negative or 0 age values detected")