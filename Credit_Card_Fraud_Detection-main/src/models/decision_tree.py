# single decision tree model - more interpretable than ensemble methods,
# good for understanding which features drive fraud predictions

# sklearn's decision tree implementation
from sklearn.tree import DecisionTreeClassifier
# config holds shared constants like USE_SMOTE
from src import config

def build():
    return DecisionTreeClassifier(
        # gini measures how mixed a node is - a node with all fraud or all legit is perfectly
        # pure (0), a 50/50 mix is maximally impure. the tree splits to reduce this as much as possible
        criterion="gini",
        # depth limit prevents the tree from memorising the training data (overfitting)
        # without this it would keep splitting until every transaction is in its own leaf
        max_depth=10,
        # no leaf node can represent fewer than 20 transactions
        # prevents tiny hyper-specific branches that only exist because of noise in the training data
        min_samples_leaf=20,
        # if SMOTE is off, tell the model to penalize fraud mistakes more heavily to offset the 99:1 class imbalance
        # if SMOTE is on, the training data is already balanced so this is not needed
        class_weight="balanced" if not config.USE_SMOTE else None,
        # fixed seed so results are reproducible across runs
        random_state=42,
    )
