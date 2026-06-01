from xgboost import XGBClassifier
from src import config

def build():

    pos_weight = config.XGB_SCALE_POS_WEIGHT if not config.USE_SMOTE else 1


    return XGBClassifier(
        # number of sequential trees to build
        n_estimators=400,

        # max depth per tree
        max_depth=6,

        # how aggressively each new tree corrects errors from the previous ones
        # lower = more conservative, less likely to overfit
        learning_rate=0.05,

        # when SMOTE is on, training data is already balanced so this is set to 1 (no weighting).
        scale_pos_weight=pos_weight,
        
        eval_metric="aucpr",
        n_jobs=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )