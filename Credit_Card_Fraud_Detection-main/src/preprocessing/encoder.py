# encoder
# handles OHE, robust scaling, and standard scaling
# fit on train only — transform applied to both train and test

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from src import config

class Encoder:

    def __init__(self):
        self._transformer = ColumnTransformer(
            transformers=[
                ("ohe",      OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.OHE_COLS),
                ("robust",   RobustScaler(),                                              config.ROBUST_SCALE_COLS),
                ("standard", StandardScaler(),                                            config.STANDARD_SCALE_COLS),
            ],
            remainder="passthrough",  # passes through any cols not explicitly listed
        )
        self._feature_names = None

    def fit_transform(self, X):
        transformed = self._transformer.fit_transform(X)
        self._feature_names = self._transformer.get_feature_names_out()
        return pd.DataFrame(transformed, columns=self._feature_names, index=X.index)

    def transform(self, X):
        if self._feature_names is None:
            raise RuntimeError("encoder has not been fit yet — call fit_transform on train first")
        transformed = self._transformer.transform(X)
        return pd.DataFrame(transformed, columns=self._feature_names, index=X.index)