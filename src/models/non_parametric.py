# %% Imports and paths
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models.utils import (
    evaluate_predictions,
    plot_calibration
)


DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "reports" / "model_results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "availability"

BASELINE_GROUPS = {
    "global": [],
    "product": ["product"],
    "outlet_type": ["outlet_type"],
    "product_outlet_type": ["product", "outlet_type"],
}
# %% Read prepared datasets

train_df = pd.read_csv(
    DATA_DIR / "train.csv"
)

val_known_df = pd.read_csv(
    DATA_DIR / "val_future_known.csv"
)

val_unseen_df = pd.read_csv(
    DATA_DIR / "val_future_unseen.csv"
)

print("Training:", train_df.shape)
print("Future-known validation:", val_known_df.shape)
print("Future-unseen validation:", val_unseen_df.shape)

print("\nTraining availability:")
print(train_df[TARGET].value_counts(normalize=True))
# %% 3. Handle missing categorical groups

MISSING_CATEGORY = "__MISSING__"


def prepare_categories(df, columns):

    result = df.copy()

    for col in columns:

        if col not in result.columns:
            raise ValueError(f"Missing required column: {col}")

        result[col] = (
            result[col]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
            .fillna(MISSING_CATEGORY)
            .astype(str)
        )

    return result


all_group_cols = sorted({
    column
    for columns in BASELINE_GROUPS.values()
    for column in columns
})

train_df = prepare_categories(train_df, all_group_cols)
val_known_df = prepare_categories(val_known_df, all_group_cols)
val_unseen_df = prepare_categories(val_unseen_df, all_group_cols)
# %% Group-frequency probability estimator

def fit_frequency_model(
    train_df,
    group_cols,
    alpha=0.0 # alpha is the smooth factor
):

    if alpha < 0:
        raise ValueError("alpha must be nonnegative.")

    global_rate = train_df[TARGET].mean()

    # Global baseline
    if not group_cols:
        return {
            "group_cols": [],
            "global_rate": global_rate,
            "group_stats": None,
            "alpha": alpha,
        }

    group_stats = (
        train_df
        .groupby(group_cols, dropna=False)
        .agg(
            n_observations=(TARGET, "size"),
            n_available=(TARGET, "sum")
        )
        .reset_index()
    )

    # Smoothed probability
    group_stats["probability"] = (
        group_stats["n_available"]
        + alpha * global_rate
    ) / (
        group_stats["n_observations"] + alpha
    )

    return {
        "group_cols": group_cols,
        "global_rate": global_rate,
        "group_stats": group_stats,
        "alpha": alpha,
    }
# %% Predict using historical group statistics

def predict_frequency_model(model, new_df):

    group_cols = model["group_cols"]
    global_rate = model["global_rate"]

    if not group_cols:
        return np.full(
            len(new_df),
            global_rate
        )

    lookup = model["group_stats"][
        group_cols + ["probability"]
    ]

    predictions = (
        new_df[group_cols]
        .merge(
            lookup,
            on=group_cols,
            how="left",
            sort=False,
            validate="many_to_one"
        )
    )

    # For combinations not found during training
    probabilities = (
        predictions["probability"]
        .fillna(global_rate)
        .to_numpy()
    )

    return probabilities
# %% Compare baseline models

ALPHA = 0.0

results = []
fitted_models = {}
validation_predictions = {}

for model_name, group_cols in BASELINE_GROUPS.items():

    print(f"\nTraining baseline: {model_name}")

    model = fit_frequency_model(
        train_df=train_df,
        group_cols=group_cols,
        alpha=ALPHA
    )

    fitted_models[model_name] = model

    # Evaluate on both validation sets
    for val_name, val_df in {
        "future_known": val_known_df,
        "future_unseen": val_unseen_df,
    }.items():

        probabilities = predict_frequency_model(
            model,
            val_df
        )

        validation_predictions[
            (model_name, val_name)
        ] = probabilities

        metrics = evaluate_predictions(
            val_df[TARGET],
            probabilities
        )

        results.append({
            "model": model_name,
            "validation": val_name,
            "alpha": ALPHA,
            **metrics
        })


results_df = pd.DataFrame(results)

print(
    results_df.sort_values(
        ["validation", "log_loss"]
    ).to_string(index=False)
)
# %%
plot_df = results_df.pivot(
    index="model",
    columns="validation",
    values="log_loss"
)

ax = plot_df.plot(
    kind="bar",
    figsize=(11, 5),
    rot=35
)

ax.set_title("Baseline Models — Validation Log Loss")
ax.set_xlabel("Model")
ax.set_ylabel("Log Loss (lower is better)")

plt.legend(title="Validation set")
plt.tight_layout()
plt.show()
# %%

plot_calibration(
    val_unseen_df[TARGET],
    validation_predictions[("product_outlet_type", "future_unseen")],
    title=f"Calibration — product_outlet_type (future_unseen)"
)
plot_calibration(
    val_known_df[TARGET],
    validation_predictions[("product_outlet_type", "future_known")],
    title=f"Calibration — product_outlet_type (future_known)"
)
# %% Evaluate all non-parametric baselines on test set

test_df = pd.read_csv(
    DATA_DIR / "test_future_unseen.csv"
)

test_df = prepare_categories(
    test_df,
    all_group_cols
)

test_results = []

for model_name, group_cols in BASELINE_GROUPS.items():

    # Fit using training observations only
    model = fit_frequency_model(
        train_df,
        group_cols,
        alpha=0
    )

    # Predict on test observations
    y_prob = predict_frequency_model(
        model,
        test_df
    )

    metrics = evaluate_predictions(
        test_df[TARGET],
        y_prob
    )

    test_results.append({
        "model": model_name,
        "alpha": 0,
        **metrics
    })

test_results_df = pd.DataFrame(test_results)

print(
    test_results_df
    .sort_values("log_loss")
    .to_string(index=False)
)
# %%
