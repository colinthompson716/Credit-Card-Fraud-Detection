# random forest model - trains many independent decision trees on random subsets of
# the data and features, then takes a majority vote across all of them.
# averaging many trees cancels out individual mistakes, giving better precision than a single tree

# sklearn's random forest implementation
from sklearn.ensemble import RandomForestClassifier
# config holds shared constants like USE_SMOTE
from src import config

def build():
    return RandomForestClassifier(
        # number of independent trees to train and average across
        n_estimators=100,
        # slightly deeper than the single decision tree (12 vs 10) because individual trees
        # in a forest are allowed to overfit a little - the ensemble average corrects for it
        max_depth=12,
        # loosened from 20 (single tree) to 10 for the same reason - individual trees can be
        # more specific since their errors are averaged out by the other 99 trees
        min_samples_leaf=10,
        # if SMOTE is off, tell the model to penalize fraud mistakes more heavily to offset the 99:1 class imbalance
        # if SMOTE is on, the training data is already balanced so this is not needed
        class_weight="balanced" if not config.USE_SMOTE else None,
        # use all available CPU cores to train trees in parallel - 100 trees would be slow otherwise
        n_jobs=-1,
        # fixed seed so results are reproducible across runs
        random_state=42,
    )
