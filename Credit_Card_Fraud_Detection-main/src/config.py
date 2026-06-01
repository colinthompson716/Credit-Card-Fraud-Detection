# preprocessing configuration
# all col names, drop lists, and constants live here

# cols to drop immediately
DROP_COLS = [
    "trans_num",    # uuid
    "city",         # redundant bc lat/long and city_pop
]

# cols not necessary after feature engineering
DROP_AFTER_ENGINEERING = [
    "trans_date_trans_time",
    "dob",
]

# target
TARGET_COL = "is_fraud"

# datetime
DATETIME_COL = "trans_date_trans_time"
DOB_COL      = "dob"

# engineered time features
TIME_FEATURES = ["trans_hour", "trans_day_of_week", "trans_month", "is_weekend", "is_night"]

# coord columns
CUSTOMER_LAT  = "lat"
CUSTOMER_LON  = "long"
MERCHANT_LAT  = "merch_lat"
MERCHANT_LON  = "merch_long"
DISTANCE_COL  = "distance_km"

# set True to drop raw coords after computing distance.
# keep False if model benefits from them (tree-based with geo splits).
DROP_COORDS_AFTER_DISTANCE = False

# high-cardinality columns (target-encode rather than OHE)
HIGH_CARDINALITY_DROP_COLS = [
    "merchant",  # 693 unique — too many for OHE, strong fraud signal
    "job",       # 163 unique — income/demographic proxy
]

# low-cardinality columns (OHE in the encoding phase, listed here for ref)
OHE_COLS = [
    "category",  # 14 unique
    "state",     # 13 unique
]

# numeric columns that need robust scaling (have heavy outliers)
ROBUST_SCALE_COLS = [
    "amt",       # max $28k vs upper fence ~$194 — StandardScaler would distort
    "city_pop",  # 65k+ outliers
]

# numeric columns safe for standard scaling
STANDARD_SCALE_COLS = [
    "lat", "long",
    "merch_lat", "merch_long",
    DISTANCE_COL,
    "age",
    "amt_log"
]

# SMOTE toggle
USE_SMOTE = True

XGB_SCALE_POS_WEIGHT = 100