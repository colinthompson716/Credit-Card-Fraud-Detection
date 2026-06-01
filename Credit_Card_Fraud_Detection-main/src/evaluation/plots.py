import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def ensure_plot_dir(path="results/plots"):
    os.makedirs(path, exist_ok=True)


def plot_model_comparison(csv_path, output_path):
    ensure_plot_dir()

    df = pd.read_csv(csv_path)

    metrics = ["precision", "recall", "f1", "roc_auc"]

    df.plot(
        x="model",
        y=metrics,
        kind="bar",
        figsize=(10, 6)
    )

    plt.title("Model Performance Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.xticks(rotation=30)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(cm, model_name, output_path):
    ensure_plot_dir()

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["legit", "fraud"]
    )

    display.plot(values_format=",d")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_threshold_curve(threshold_results, model_name, output_path):
    ensure_plot_dir()

    plt.figure(figsize=(8, 5))
    plt.plot(threshold_results["threshold"], threshold_results["precision"], label="Precision")
    plt.plot(threshold_results["threshold"], threshold_results["recall"], label="Recall")
    plt.plot(threshold_results["threshold"], threshold_results["f1"], label="F1")

    plt.title(f"{model_name} Threshold Tuning")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_feature_importance(csv_path, model_name, output_path):
    df = pd.read_csv(csv_path).head(10)

    plt.figure(figsize=(8, 5))
    plt.barh(df["feature"], df["importance"])
    plt.gca().invert_yaxis()

    plt.title(f"{model_name} Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

from sklearn.metrics import roc_curve, roc_auc_score


def plot_roc_comparison(model_probas, y_true, output_path):
    ensure_plot_dir()

    plt.figure(figsize=(9, 6))

    for model_name, y_proba in model_probas.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        plt.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC={auc:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Guessing"
    )

    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()