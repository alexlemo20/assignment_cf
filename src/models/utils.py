
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)

from sklearn.calibration import calibration_curve


NOMINAL_COLS = [
    "product",
    "product_type",
    "outlet_type",
]

ORDINAL_COLS = [
    "sales_performance_indicator",
    "outlet_volume_potentiality_ord",
    "outlet_category_volume_potentiality_ord",
    "outlet_satisfaction",
    "price_range_indicator",
]

NUMERIC_COLS = [
    "consumer_sentiment",
    "popularity_with_consumers",
    "consumer_disposable_income_index",
    "consumer_spending_per_capita_on_food_beverages",
    "surrounding_traffic",
    "touristic_point_of_interest",
    "number_of_competing_outlets",
]

BINARY_COLS = [
    "cooler_presence_binary",
]

CALENDAR_COLS = [
    "month",
    "day_of_week",
    "day_of_month",
]

HIGH_MISSING_COLS = [
    "outlet_satisfaction",
    "price_range_indicator",
    "consumer_sentiment",
    "popularity_with_consumers",
]

def evaluate_predictions(y_true, y_prob, threshold=0.5):

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    if not np.isfinite(y_prob).all():
        raise ValueError("Predictions contain invalid values.")

    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("Probabilities must be between 0 and 1.")

    y_pred = (y_prob >= threshold).astype(int)

    results = {
        "n_samples": len(y_true),
        "availability_rate": y_true.mean(),
        "mean_predicted_probability": y_prob.mean(),
        "log_loss": log_loss(
            y_true,
            y_prob,
            labels=[0, 1]
        ),
        "brier_score": brier_score_loss(
            y_true,
            y_prob
        ),
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),
    }

    if len(np.unique(y_true)) == 2:

        results["roc_auc"] = roc_auc_score(
            y_true,
            y_prob
        )

        # Positive class here is UNAVAILABILITY
        results["pr_auc_unavailable"] = (
            average_precision_score(
                1 - y_true,
                1 - y_prob
            )
        )

    else:
        results["roc_auc"] = np.nan
        results["pr_auc_unavailable"] = np.nan

    return results

def plot_calibration(y_true, y_prob, title="Calibration"):

    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=10,
        strategy="quantile"
    )

    plt.figure(figsize=(6, 5))

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        color="black",
        label="Perfect calibration"
    )

    plt.plot(
        prob_pred,
        prob_true,
        marker="o",
        label="Model"
    )

    plt.xlabel("Mean predicted availability")
    plt.ylabel("Observed availability")
    plt.title(title)

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def prepare_model_features(
    dataset,
    numerical_features,
    categorical_features,
):
    columns = numerical_features + categorical_features
    X = dataset[columns].copy()

    for col in numerical_features:
        X[col] = (
            pd.to_numeric(X[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )

    for col in categorical_features:
        X[col] = (
            X[col]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
            .astype(object)
        )

        X[col] = X[col].where(X[col].notna(), np.nan)

    return X


def build_mlp_preprocessor(
    numerical_features,
    categorical_features,
):
    transformers = []

    if numerical_features:
        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ])

        transformers.append((
            "numeric",
            numeric_pipeline,
            numerical_features,
        ))

    if categorical_features:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    keep_empty_features=True,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ])

        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_features,
        ))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
    )

def add_prediction_errors(df):
    """Add per-observation squared and log errors to a prediction table."""
    result = df.copy()
    result["squared_error"] = (result["y_true"] - result["y_prob"]) ** 2
    p = np.clip(result["y_prob"], 1e-15, 1 - 1e-15)
    result["log_error"] = -(
        result["y_true"] * np.log(p)
        + (1 - result["y_true"]) * np.log(1 - p)
    )
    return result


def summarize_prediction_errors(df, feature):
    """Summarize prediction errors by a categorical feature."""
    return (
        df.groupby(feature, dropna=False)
        .agg(
            n_observations=("y_true", "size"),
            actual_availability=("y_true", "mean"),
            predicted_availability=("y_prob", "mean"),
            brier_score=("squared_error", "mean"),
            log_loss=("log_error", "mean"),
        )
        .sort_values("log_loss", ascending=False)
    )
