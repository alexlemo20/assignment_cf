# %% Imports and setup
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from types import SimpleNamespace

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

from src.models.utils import (
    evaluate_predictions,
    plot_calibration,
    ORDINAL_COLS,
    NUMERIC_COLS,
    BINARY_COLS,
    CALENDAR_COLS,
    NOMINAL_COLS,
    prepare_model_features,
    add_prediction_errors,
    summarize_prediction_errors,
)


DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "reports" / "model_results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "availability"
SEED = 42
# %%
train_df = pd.read_csv(DATA_DIR / "train.csv")

val_known_df = pd.read_csv(
    DATA_DIR / "val_future_known.csv"
)

val_unseen_df = pd.read_csv(
    DATA_DIR / "val_future_unseen.csv"
)

print("Training:", train_df.shape)
print("Validation known:", val_known_df.shape)
print("Validation unseen:", val_unseen_df.shape)

print("\nTraining target distribution:")
print(train_df[TARGET].value_counts(normalize=True))
# %% Feature configuration
NUMERICAL_FEATURES = (
    ORDINAL_COLS
    + NUMERIC_COLS
    + BINARY_COLS
    + CALENDAR_COLS
)

CATEGORICAL_FEATURES = NOMINAL_COLS

SELECTED_FEATURES = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)


print("Numerical features:", NUMERICAL_FEATURES)
print("Categorical features:", CATEGORICAL_FEATURES)

# %% Model configuration
args = SimpleNamespace(
    experiment_name="xgboost_default",
    n_estimators=500,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    early_stopping_rounds=30,
    numeric_missing="median", # native
)

def prepare_features(dataset):
    return prepare_model_features(
        dataset, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
    )


X_train = prepare_features(train_df)
X_val_known = prepare_features(val_known_df)
X_val_unseen = prepare_features(val_unseen_df)

y_train = train_df[TARGET].astype(int)
y_val_known = val_known_df[TARGET].astype(int)
y_val_unseen = val_unseen_df[TARGET].astype(int)

print(X_train.dtypes)
# %%
def build_preprocessor(numeric_missing="median"):

    transformers = []

    # Numerical and ordinal features
    if numeric_missing == "median":

        numeric_pipeline = SimpleImputer(
            strategy="median",
            keep_empty_features=True
        )

        transformers.append((
            "numeric",
            numeric_pipeline,
            NUMERICAL_FEATURES
        ))

    elif numeric_missing == "native":

        # No imputation; retain NaN for XGBoost
        transformers.append((
            "numeric",
            "passthrough",
            NUMERICAL_FEATURES
        ))

    else:
        raise ValueError(
            f"Unsupported missing strategy: {numeric_missing}"
        )

    # Categorical features
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="__MISSING__",
                keep_empty_features=True
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ])

    transformers.append((
        "categorical",
        categorical_pipeline,
        CATEGORICAL_FEATURES
    ))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0
    )
preprocessor = build_preprocessor(
    numeric_missing=args.numeric_missing
)

X_train_processed = preprocessor.fit_transform(X_train)

X_val_known_processed = preprocessor.transform(X_val_known)
X_val_unseen_processed = preprocessor.transform(X_val_unseen)

print("Training shape:", X_train_processed.shape)
print("Validation known shape:", X_val_known_processed.shape)
print("Validation unseen shape:", X_val_unseen_processed.shape)

assert (
    X_train_processed.shape[1]
    == X_val_unseen_processed.shape[1]
)
# %% Train XGBoost
classifier = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",

    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    learning_rate=args.learning_rate,
    min_child_weight=args.min_child_weight,
    subsample=args.subsample,
    colsample_bytree=args.colsample_bytree,
    reg_lambda=args.reg_lambda,

    early_stopping_rounds=args.early_stopping_rounds,

    tree_method="hist",
    random_state=SEED,
    n_jobs=-1,
)

classifier.fit(
    X_train_processed,
    y_train,

    eval_set=[
        (X_train_processed, y_train),
        (X_val_unseen_processed, y_val_unseen),
    ],

    verbose=False,
)

print("Model trained successfully.")

print("Best boosting iteration:", classifier.best_iteration)
print("Best validation log loss:", classifier.best_score)
# %% XGBoost learning curves
history = classifier.evals_result()

train_loss = history["validation_0"]["logloss"]
val_loss = history["validation_1"]["logloss"]

iterations = range(1, len(train_loss) + 1)

plt.figure(figsize=(10, 5))

plt.plot(
    iterations,
    train_loss,
    label="Training Log Loss"
)

plt.plot(
    iterations,
    val_loss,
    label="Unseen Validation Log Loss"
)

plt.axvline(
    classifier.best_iteration + 1,
    linestyle="--",
    color="black",
    label="Best iteration"
)

plt.xlabel("Boosting iteration")
plt.ylabel("Log Loss")
plt.title("XGBoost — Training and Validation Loss")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %% Evaluate validation performance
validation_sets = {
    "future_known": (
        X_val_known_processed,
        y_val_known,
        val_known_df
    ),
    "future_unseen": (
        X_val_unseen_processed,
        y_val_unseen,
        val_unseen_df
    ),
}

results = []
prediction_tables = {}

for name, (X_val, y_val, original_df) in validation_sets.items():

    y_prob = classifier.predict_proba(X_val)[:, 1]

    metrics = evaluate_predictions(
        y_val,
        y_prob
    )

    results.append({
        "model": "xgboost",
        "experiment": args.experiment_name,
        "validation": name,
        "best_iteration": classifier.best_iteration,
        **metrics
    })

    prediction_tables[name] = pd.DataFrame({
        "outlet": original_df["outlet"].to_numpy(),
        "product": original_df["product"].to_numpy(),
        "survey_date": original_df["survey_date"].to_numpy(),
        "y_true": y_val.to_numpy(),
        "y_prob": y_prob
    })

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))
# %% Probability calibration
for name, predictions in prediction_tables.items():

    plot_calibration(
        predictions["y_true"],
        predictions["y_prob"],
        title=f"XGBoost — {args.experiment_name} ({name})"
    )
# %% XGBoost feature importance
feature_names = preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": classifier.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print(importance_df.head(25).to_string(index=False))

plt.figure(figsize=(10, 7))

top_features = (
    importance_df
    .head(20)
    .sort_values("importance")
)

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.xlabel("Feature importance")
plt.title("XGBoost — Feature Importance")
plt.tight_layout()
plt.show()
# %% Error analysis by product and outlet type

error_df = val_unseen_df.copy()

error_df["y_true"] = (
    prediction_tables["future_unseen"]["y_true"].to_numpy()
)

error_df["y_prob"] = (
    prediction_tables["future_unseen"]["y_prob"].to_numpy()
)

error_df = add_prediction_errors(error_df)

for feature in ["product", "outlet_type"]:

    summary = summarize_prediction_errors(error_df, feature)

    print(f"\nError analysis by {feature}")
    print(summary.to_string())
# %% Final XGBoost evaluation on test set

test_df = pd.read_csv(
    DATA_DIR / "test_future_unseen.csv"
)

X_test = prepare_features(test_df)
y_test = test_df[TARGET].astype(int)

# Apply fitted training preprocessing, without refitting
X_test_processed = preprocessor.transform(X_test)

# Predict probabilities using the trained XGBoost classifier
y_prob_test = classifier.predict_proba(
    X_test_processed
)[:, 1]

test_metrics = evaluate_predictions(
    y_test,
    y_prob_test
)

test_results_df = pd.DataFrame([{
    "model": "xgboost",
    "experiment": args.experiment_name,
    "dataset": "test_future_unseen",
    "best_iteration": classifier.best_iteration,
    **test_metrics,
}])

print(test_results_df.to_string(index=False))
# %%
