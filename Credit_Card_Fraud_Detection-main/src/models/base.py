# base model wrapper
# FraudClassifier wraps any sklearn-compatible estimator
# adds threshold tuning + logging

import numpy as np
import pandas as pd

class FraudClassifier:

    def __init__(self, estimator, threshold: float = 0.5):
        self.estimator = estimator
        self.threshold = threshold

    # public API

    def fit(self, X, y):
        self._log(f"training {self._name()} on {X.shape[0]:,} samples, {X.shape[1]} features")
        self.estimator.fit(X, y)
        self._log("training complete")
        return self
    
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)[:,1]
    
    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)
    
    # helpers

    def _name(self):
        return type(self.estimator).__name__
    
    def _log(self, message):
        print(f"[{self._name()}] {message}")