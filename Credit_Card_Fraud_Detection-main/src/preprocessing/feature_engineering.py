# feature engineering transformations

import numpy as np
import pandas as pd

from src import config

# datetime features
def add_time_features(df):
    col = df[config.DATETIME_COL]

    df["trans_hour"]        = col.dt.hour
    df["trans_day_of_week"] = col.dt.dayofweek 
    df["trans_month"]       = col.dt.month
    df["is_weekend"]        = col.dt.dayofweek.isin([5, 6]).astype(int)
    df["is_night"]          = ((col.dt.hour >= 22) | (col.dt.hour <= 6)).astype(int)

    return df

# use datetime of transaction and dob to calculate customer age
def add_age(df):
    delta = df[config.DATETIME_COL] - df[config.DOB_COL]
    df["age"] = delta.dt.days / 365.25
    return df

# log transform of transaction amount, reduces the impact of extreme outliers (e.g. $28k transactions)
def add_amt_log(df):
    df["amt_log"] = np.log1p(df["amt"])
    return df

# geographic features

# haversine distance formula — computes straight line distance
# between two lat/lon coordinates on the surface of the Earth
def _haversine(lat1, lon1, lat2, lon2):
    R = 6_371.0  # Earth radius, km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))

# compute straight line distance from customer's home location to merchant location, EDA showed this feature has a near zero correlation with fraud but kept
# for the models to decide, tree-based models can ignore it if not useful
def add_distance(df):
    df[config.DISTANCE_COL] = _haversine(
        df[config.CUSTOMER_LAT], df[config.CUSTOMER_LON],
        df[config.MERCHANT_LAT], df[config.MERCHANT_LON]
    )
    return df