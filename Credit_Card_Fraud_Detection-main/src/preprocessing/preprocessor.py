# preprocessor
# orchestrates all preprocessing stages in order

import pandas as pd

from src import config
from . import cleaning, feature_engineering

class Preprocessor:
    # orchestrates full preprocessing pipeline
    # encoding and scaling live in next pipeline stage so that
    # train/test split happens in between, which prevents data leakage

    # 1. clean()    - drop useless columns, cast types
    # 2. engineer() - add time features, age, distance
    # 3. finalize() - drop raw source columns, validate output

    def __init__(self):
        pass

    # public API

    def run(self, df):
        df = self.clean(df)
        df = self.engineer(df)
        df = self.finalize(df)

        return df
    
    def clean(self, df):
        df = df.copy()
        self._log(f"input shape: {df.shape}")

        df = cleaning.drop_initial_cols(df)
        self._log(f"after dropping initial cols: {df.shape}")

        df = cleaning.cast_datetimes(df)
        self._log("datetime columns cast successfully")

        return df
    
    def engineer(self, df):
        df = feature_engineering.add_time_features(df)
        self._log(f"time features added: {config.TIME_FEATURES}")

        df = feature_engineering.add_age(df)
        self._log("age feature added")

        df = feature_engineering.add_amt_log(df)
        self._log("amt_log feature added (log transform of transaction amount)")

        df = feature_engineering.add_distance(df)
        self._log(f"distance feature added: {config.DISTANCE_COL}")

        return df
    
    def finalize(self, df):
        df = cleaning.drop_engineered_sources(df)
        self._log(f"after dropping engineered sources: {df.shape}")

        df = cleaning.drop_coords_if_configured(df)

        cleaning.validate(df)
        self._log(f"validation passed. final shape: {df.shape}")
        self._log(f"columns ready for encoding/scaling: {sorted(df.columns.tolist())}")

        return df

    # helpers

    def _log(self, message):
        print(f"[preprocessor] {message}")
