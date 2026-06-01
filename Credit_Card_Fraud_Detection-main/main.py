from src.preprocessing.preprocessor import Preprocessor
from src.preprocessing.encoder import Encoder
from src.models.base import FraudClassifier
from src.models import logistic, decision_tree, random_forest, xgboost_model
from src import config
from src.evaluation import evaluator, plots

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd


def main():
    df = pd.read_csv("data/raw/credit_card_frauds.csv")
    print(f"[data] loaded: {df.shape}")

    preprocessor = Preprocessor()
    df_processed = preprocessor.run(df)

    X = df_processed.drop(columns=[config.TARGET_COL])
    y = df_processed[config.TARGET_COL]

    # drop high-cardinality categorical columns
    # target encoding was considered but left out to keep the pipeline simpler
    # and avoid leakage risk
    X = X.drop(columns=config.HIGH_CARDINALITY_DROP_COLS)

    # first split off the final test set
    # this test set stays untouched until final evaluation
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # split the remaining training data into train and validation sets
    # final proportions: 60% train, 20% validation, 20% test
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        stratify=y_train_full,
        random_state=42,
    )

    print(f"[split] train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")
    print(
        f"[split] fraud rate — "
        f"train: {y_train.mean():.4f}, "
        f"val: {y_val.mean():.4f}, "
        f"test: {y_test.mean():.4f}"
    )

    encoder = Encoder()
    X_train = encoder.fit_transform(X_train)
    X_val = encoder.transform(X_val)
    X_test = encoder.transform(X_test)

    print(f"[encoder] train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    # SMOTE is applied only to the training data, never validation or test.
    if config.USE_SMOTE:
        from imblearn.over_sampling import SMOTE

        X_train, y_train = SMOTE(random_state=67).fit_resample(X_train, y_train)
        print(f"[smote] resampled train: {X_train.shape}, fraud rate: {y_train.mean():.4f}")

    models = {
        "logistic": FraudClassifier(logistic.build()),
        "decision_tree": FraudClassifier(decision_tree.build()),
        "random_forest": FraudClassifier(random_forest.build()),
        "xgboost": FraudClassifier(xgboost_model.build()),
    }

    baseline_results = []
    tuned_results = []
    model_probas = {}

    for name, model in models.items():
        print(f"\n{'=' * 50}")
        print(f"MODEL: {name.upper()}")
        print(f"{'=' * 50}")

        model.fit(X_train, y_train)

        if hasattr(model.estimator, "feature_importances_"):
            importances = pd.DataFrame({
                "feature": X_train.columns,
                "importance": model.estimator.feature_importances_,
            }).sort_values(by="importance", ascending=False)

            importance_path = f"results/metrics/{name}_feature_importance.csv"
            importances.to_csv(importance_path, index=False)

            print("\n--- Top 10 Feature Importances ---")
            print(importances.head(10))

            plots.plot_feature_importance(
                importance_path,
                name,
                f"results/plots/{name}_feature_importance.png",
            )

        # validation probabilities are used for threshold tuning
        # test probabilities are used only for final evaluation and ROC comparison.
        val_proba = model.predict_proba(X_val)
        test_proba = model.predict_proba(X_test)
        model_probas[name] = test_proba

        # baseline evaluation using the default 0.5 threshold on the test set
        baseline_threshold = 0.5
        baseline_pred = (test_proba >= baseline_threshold).astype(int)

        baseline_metrics = evaluator.evaluate_model(
            name=name,
            y_true=y_test,
            y_pred=baseline_pred,
            y_proba=test_proba,
        )
        baseline_metrics["threshold"] = baseline_threshold
        baseline_results.append(baseline_metrics)

        # tune threshold on validation set only
        threshold_results = evaluator.tune_threshold(
            name=name,
            y_true=y_val,
            y_proba=val_proba,
        )

        threshold_results["score"] = (
            0.7 * threshold_results["recall"]
            + 0.3 * threshold_results["precision"]
        )

        best_row = threshold_results.sort_values(
            by="score",
            ascending=False,
        ).iloc[0]

        best_threshold = best_row["threshold"]

        # apply the validation-selected threshold to the test set.
        tuned_pred = (test_proba >= best_threshold).astype(int)

        tuned_metrics = evaluator.evaluate_model(
            name=name,
            y_true=y_test,
            y_pred=tuned_pred,
            y_proba=test_proba,
        )
        tuned_metrics["threshold"] = best_threshold
        tuned_metrics["validation_score"] = best_row["score"]
        tuned_results.append(tuned_metrics)

        print(f"\nBaseline threshold: {baseline_threshold}")
        print(f"Tuned threshold selected on validation set: {best_threshold:.2f}")
        print(f"Validation score: {best_row['score']:.4f}")

        print("\n--- Tuned Classification Report on Test Set ---")
        print(classification_report(y_test, tuned_pred, target_names=["legit", "fraud"]))

        print("--- Tuned Confusion Matrix on Test Set ---")
        cm = confusion_matrix(y_test, tuned_pred)

        plots.plot_confusion_matrix(
            cm,
            name,
            f"results/plots/{name}_confusion_matrix.png",
        )

        plots.plot_threshold_curve(
            threshold_results,
            name,
            f"results/plots/{name}_threshold_curve.png",
        )

        print(f"  True Negatives:  {cm[0][0]:,}")
        print(f"  False Positives: {cm[0][1]:,}")
        print(f"  False Negatives: {cm[1][0]:,}")
        print(f"  True Positives:  {cm[1][1]:,}")

    evaluator.save_results(
        baseline_results,
        "results/metrics/baseline_model_comparison.csv",
        title="BASELINE MODEL COMPARISON",
    )

    evaluator.save_results(
        tuned_results,
        "results/metrics/tuned_model_comparison.csv",
        title="TUNED MODEL COMPARISON",
    )

    plots.plot_model_comparison(
        "results/metrics/baseline_model_comparison.csv",
        "results/plots/baseline_model_comparison.png",
    )

    plots.plot_model_comparison(
        "results/metrics/tuned_model_comparison.csv",
        "results/plots/tuned_model_comparison.png",
    )

    plots.plot_roc_comparison(
        model_probas,
        y_test,
        "results/plots/roc_curve_comparison.png",
    )


if __name__ == "__main__":
    main()