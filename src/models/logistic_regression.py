# %% Imports and paths
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from src.models.utils import (
    evaluate_predictions,
    plot_calibration,
    ORDINAL_COLS,
    NUMERIC_COLS,
    BINARY_COLS,
    CALENDAR_COLS,
    NOMINAL_COLS,
    HIGH_MISSING_COLS,
    prepare_model_features,
    add_prediction_errors,
    summarize_prediction_errors,
)


DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "reports" / "model_results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "availability"
# %% Feature configuration
# Original categorical versions used for one-hot encoding
ORDINAL_ONEHOT_COLS = [
    "sales_performance_indicator",
    "outlet_volume_potentiality",
    "outlet_category_volume_potentiality",
    "outlet_satisfaction",
    "price_range_indicator",
]

FEATURE_SETS = {
    "all": "all",
    "no_high_missing": "no_high_missing",
}

# %% Experiment arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--features",
    choices=list(FEATURE_SETS),
    default="all",
)

parser.add_argument(
    "--ordinal-encoding",
    choices=["ordinal", "onehot"],
    default="ordinal",
)

parser.add_argument(
    "--numeric-imputation",
    choices=["median", "mean"],
    default="median",
)

parser.add_argument(
    "--categorical-imputation",
    choices=["missing", "most_frequent"],
    default="missing",
)

parser.add_argument(
    "--penalty",
    choices=["l1", "l2", "elasticnet"],
    default="l2",
)

parser.add_argument(
    "--experiment-name",
    type=str,
    default="logreg_default",
)

# Compatible with interactive notebook execution
# args, _ = parser.parse_known_args()
# print(vars(args))
from types import SimpleNamespace
args = SimpleNamespace(
    features="all", # "no_high_missing",
    ordinal_encoding="ordinal",
    numeric_imputation="median",
    categorical_imputation="missing",
    penalty="l2",
    experiment_name="logreg_default",
)
# %%
def select_features(feature_set, ordinal_encoding):

    nominal_features = NOMINAL_COLS.copy()
    numeric_features = NUMERIC_COLS + BINARY_COLS

    if ordinal_encoding == "ordinal":
        ordinal_features = ORDINAL_COLS + CALENDAR_COLS
    else:
        ordinal_features = ORDINAL_ONEHOT_COLS + CALENDAR_COLS

    if feature_set == "no_high_missing":

        ordinal_features = [
            c for c in ordinal_features
            if c not in HIGH_MISSING_COLS
        ]

        numeric_features = [
            c for c in numeric_features
            if c not in HIGH_MISSING_COLS
        ]

    selected_features = (
        nominal_features
        + ordinal_features
        + numeric_features
    )

    assert len(selected_features) == len(set(selected_features)), (
        "Duplicate feature names found."
    )

    return {
        "nominal": nominal_features,
        "ordinal": ordinal_features,
        "numeric": numeric_features,
        "all": selected_features,
    }


feature_config = select_features(
    feature_set=args.features,
    ordinal_encoding=args.ordinal_encoding,
)

selected_features = feature_config["all"]

print("Selected encoding:", args.ordinal_encoding)
print("Selected features:", selected_features)

#%% Read data
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
for name, dataset in {
    "train": train_df,
    "val_known": val_known_df,
    "val_unseen": val_unseen_df,
}.items():

    missing = set(selected_features) - set(dataset.columns)

    if missing:
        raise ValueError(
            f"{name} is missing columns: {sorted(missing)}"
        )

# %% Prepare feature representations
def prepare_features(dataset, feature_config):
    numerical_inputs = feature_config["numeric"] + (
        feature_config["ordinal"] if args.ordinal_encoding == "ordinal" else []
    )
    categorical_inputs = feature_config["nominal"] + (
        feature_config["ordinal"] if args.ordinal_encoding == "onehot" else []
    )
    return prepare_model_features(dataset, numerical_inputs, categorical_inputs)


X_train = prepare_features(train_df, feature_config)
X_val_known = prepare_features(val_known_df, feature_config)
X_val_unseen = prepare_features(val_unseen_df, feature_config)

y_train = train_df[TARGET].astype(int)
y_val_known = val_known_df[TARGET].astype(int)
y_val_unseen = val_unseen_df[TARGET].astype(int)

print(X_train.dtypes)
print("\nTraining shape:", X_train.shape)

# %% Build preprocessing pipeline
def build_preprocessor(feature_config, args):

    nominal_features = feature_config["nominal"]

    ordinal_features = feature_config["ordinal"]

    numeric_features = [
        c for c in feature_config["numeric"]
        if c not in BINARY_COLS
    ]

    binary_features = [
        c for c in BINARY_COLS
        if c in feature_config["numeric"]
    ]

    if args.ordinal_encoding == "ordinal":
        numeric_features = (
            numeric_features + ordinal_features
        )
        categorical_features = nominal_features

    else:
        categorical_features = (
            nominal_features + ordinal_features
        )

    # Numeric variables: impute, then standardize
    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy=args.numeric_imputation,
                add_indicator=True,
                keep_empty_features=True,
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    # Categorical missing-value treatment
    if args.categorical_imputation == "missing":

        categorical_imputer = SimpleImputer(
            strategy="constant",
            fill_value="__MISSING__",
            keep_empty_features=True,
        )

    else:

        categorical_imputer = SimpleImputer(
            strategy="most_frequent",
            keep_empty_features=True,
        )

    categorical_pipeline = Pipeline([
        ("imputer", categorical_imputer),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
            )
        )
    ])

    # Binary variables: retain 0/1 interpretation
    binary_pipeline = SimpleImputer(
        strategy="most_frequent",
        add_indicator=True,
        keep_empty_features=True,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
            ("binary", binary_pipeline, binary_features),
        ],
        remainder="drop"
    )

    return preprocessor

# %% Build and fit Logistic Regression
preprocessor = build_preprocessor(
    feature_config,
    args
)

classifier = LogisticRegression(
    penalty=args.penalty,
    max_iter=5000,
    random_state=42,
    # solver="saga",
    # l1_ratio=0.5,
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", classifier),
])

model.fit(X_train, y_train)

print("Model fitted successfully.")

print(
    "Number of coefficients:",
    model.named_steps["classifier"].coef_.shape[1]
)
# %% Evaluate on both validation sets
validation_sets = {
    "future_known": (X_val_known, y_val_known, val_known_df),
    "future_unseen": (X_val_unseen, y_val_unseen, val_unseen_df),
}

results = []
prediction_tables = {}

for name, (X_val, y_val, original_df) in validation_sets.items():

    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = evaluate_predictions(
        y_val,
        y_prob
    )

    results.append({
        "model": "logistic_regression",
        "experiment": args.experiment_name,
        "validation": name,
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
pred_df = prediction_tables["future_unseen"]

plot_calibration(
    pred_df["y_true"],
    pred_df["y_prob"],
    title=f"Logistic Regression — {args.experiment_name}"
)
pred_df = prediction_tables["future_known"]

plot_calibration(
    pred_df["y_true"],
    pred_df["y_prob"],
    title=f"Logistic Regression — {args.experiment_name}"
)

# %% Inspect model coefficients
feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

coefficients = (
    model.named_steps["classifier"]
    .coef_[0]
)

coefficients_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
})

coefficients_df["abs_coefficient"] = (
    coefficients_df["coefficient"].abs()
)

coefficients_df = coefficients_df.sort_values(
    "abs_coefficient",
    ascending=False
)

print(coefficients_df.head(25).to_string(index=False))
# %% Error analysis by product and outlet type
error_df = val_unseen_df.copy()

error_df["y_prob"] = prediction_tables["future_unseen"]["y_prob"]
error_df["y_true"] = prediction_tables["future_unseen"]["y_true"]

error_df = add_prediction_errors(error_df)

for feature in ["product", "outlet_type"]:

    summary = summarize_prediction_errors(error_df, feature)

    print(f"\nError analysis by {feature}")
    print(summary.to_string())
# %% High-confidence prediction errors

error_df["absolute_error"] = (
    error_df["y_true"] - error_df["y_prob"]
).abs()

worst_predictions = error_df.sort_values(
    "absolute_error",
    ascending=False
)

print(
    worst_predictions[
        [
            "outlet",
            "product",
            "outlet_type",
            "survey_date",
            "y_true",
            "y_prob",
            "absolute_error",
        ]
    ].head(20).to_string(index=False)
)
# %% Evaluate final model on unseen test set
test_df = pd.read_csv(
    DATA_DIR / "test_future_unseen.csv"
)
X_test = prepare_features(
    test_df,
    feature_config
)

y_test = test_df[TARGET].astype(int)

y_prob_test = model.predict_proba(X_test)[:, 1]

test_metrics = evaluate_predictions(
    y_test,
    y_prob_test
)

test_results_df = pd.DataFrame([{
    "model": "logistic_regression",
    "experiment": args.experiment_name,
    "dataset": "test_future_unseen",
    **test_metrics
}])

print(test_results_df.to_string(index=False))
# %%
