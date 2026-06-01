import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

def evaluate_model(name, y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "true_negatives": tn,
    }


def save_results(results, path, title):
    df = pd.DataFrame(results)

    df = df.sort_values(
        by=["f1", "recall"],
        ascending=False
    )

    print(f"\n=== {title} ===")
    print(df)

    df.to_csv(path, index=False)


def tune_threshold(name, y_true, y_proba):
    thresholds = [i / 100 for i in range(1, 100)]
    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        metrics = evaluate_model(
            name=name,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba
        )

        metrics["threshold"] = threshold
        results.append(metrics)

    return pd.DataFrame(results)