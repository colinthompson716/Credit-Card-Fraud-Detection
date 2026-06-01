# logistic regression model - used as a simple baseline to compare other models against

# sklearn's logistic regression implementation
from sklearn.linear_model import LogisticRegression
# config holds shared constants like USE_SMOTE
from src import config

def build():
    return LogisticRegression(
        # allow enough iterations for the solver to converge on large datasets
        max_iter=1000,
        # if SMOTE is off, tell the model to penalize fraud mistakes more heavily to offset the 99:1 class imbalance
        # if SMOTE is on, the training data is already balanced so this is not needed
        class_weight="balanced" if not config.USE_SMOTE else None,
        # fixed seed so results are reproducible across runs
        random_state=67
    )
