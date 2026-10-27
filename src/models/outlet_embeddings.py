# %% Imports and setup

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from copy import deepcopy
from tqdm.auto import tqdm

import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA

from src.models.utils import (
    evaluate_predictions,
    plot_calibration,
    ORDINAL_COLS,
    NUMERIC_COLS,
    BINARY_COLS,
    CALENDAR_COLS,
    prepare_model_features,
    build_mlp_preprocessor,
)

from src.models.MLP import OutletEmbeddingMLP


EXPERIMENT_NAME = "outlet_embeddings_default"
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "reports" / "model_results"
EXPERIMENT_DIR = RESULTS_DIR / EXPERIMENT_NAME

TARGET = "availability"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
# %% Load datasets
train_df = pd.read_csv(DATA_DIR / "train.csv")

val_known_df = pd.read_csv(
    DATA_DIR / "val_future_known.csv"
)

val_unseen_df = pd.read_csv(
    DATA_DIR / "val_future_unseen.csv"
)

datasets = {
    "train": train_df,
    "future_known": val_known_df,
    "future_unseen": val_unseen_df,
}
# %% Feature groups
OUTLET_NUMERIC = (
    ORDINAL_COLS
    + NUMERIC_COLS
    + BINARY_COLS
)

OUTLET_CATEGORICAL = [
    "outlet_type",
]

CONTEXT_NUMERIC = CALENDAR_COLS

CONTEXT_CATEGORICAL = [
    "product",
    "product_type",
    "outlet_category_volume_potentiality_ord",
]

OUTLET_FEATURES = (
    OUTLET_NUMERIC + OUTLET_CATEGORICAL
)
OUTLET_FEATURES.remove("outlet_category_volume_potentiality_ord")

CONTEXT_FEATURES = (
    CONTEXT_NUMERIC + CONTEXT_CATEGORICAL
)
print("Outlet features:", OUTLET_FEATURES)
print("Context features:", CONTEXT_FEATURES)

# %% Preprocess outlet and context features
outlet_preprocessor = build_mlp_preprocessor(
    OUTLET_NUMERIC,
    OUTLET_CATEGORICAL,
)

context_preprocessor = build_mlp_preprocessor(
    CONTEXT_NUMERIC,
    CONTEXT_CATEGORICAL,
)

prepared = {}

for name, df in datasets.items():

    X_outlet = prepare_model_features(
        df,
        OUTLET_NUMERIC,
        OUTLET_CATEGORICAL,
    )

    X_context = prepare_model_features(
        df,
        CONTEXT_NUMERIC,
        CONTEXT_CATEGORICAL,
    )

    prepared[name] = {
        "outlet": X_outlet,
        "context": X_context,
        "y": df[TARGET].to_numpy(dtype=np.float32),
    }


# Fit transformations ONLY on training data
outlet_preprocessor.fit(
    prepared["train"]["outlet"]
)

context_preprocessor.fit(
    prepared["train"]["context"]
)


processed = {}

for name, values in prepared.items():

    processed[name] = {
        "outlet": np.asarray(
            outlet_preprocessor.transform(values["outlet"]),
            dtype=np.float32,
        ),
        "context": np.asarray(
            context_preprocessor.transform(values["context"]),
            dtype=np.float32,
        ),
        "y": values["y"],
    }


OUTLET_INPUT_DIM = processed["train"]["outlet"].shape[1]
CONTEXT_INPUT_DIM = processed["train"]["context"].shape[1]

print("Outlet input dimension:", OUTLET_INPUT_DIM)
print("Context input dimension:", CONTEXT_INPUT_DIM)
# %% Initialize model and DataLoader

train = processed["train"]
val = processed["future_unseen"]

train_dataset = TensorDataset(
    torch.tensor(train["outlet"]),
    torch.tensor(train["context"]),
    torch.tensor(train["y"]).reshape(-1, 1),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)

X_val_outlet = torch.tensor(
    val["outlet"],
    device=device,
)

X_val_context = torch.tensor(
    val["context"],
    device=device,
)

y_val_tensor = torch.tensor(
    val["y"],
    device=device,
).reshape(-1, 1)


model = OutletEmbeddingMLP(
    outlet_input_dim=OUTLET_INPUT_DIM,
    context_input_dim=CONTEXT_INPUT_DIM,
    embedding_dim=4,
    encoder_hidden_dim=16,
    prediction_hidden_dim=16,
    dropout=0.1,
).to(device)

n_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(model)
print("Trainable parameters:", n_parameters)
# %% Train outlet embedding model
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-3,
)

MAX_EPOCHS = 200
PATIENCE = 15

best_loss = float("inf")
best_epoch = 0
best_weights = None
epochs_without_improvement = 0

history = {
    "train_loss": [],
    "val_loss": [],
}

for epoch in tqdm(
    range(MAX_EPOCHS),
    leave=False
    ):

    model.train()
    running_loss = 0.0

    for X_outlet, X_context, y_batch  in tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{MAX_EPOCHS}",
        leave=False
        ):

        X_outlet = X_outlet.to(device)
        X_context = X_context.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_outlet, X_context)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(y_batch)

    train_loss = running_loss / len(train_dataset)

    model.eval()

    with torch.no_grad():
        val_logits = model(
            X_val_outlet,
            X_val_context,
        )

        val_loss = criterion(
            val_logits,
            y_val_tensor,
        ).item()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    if val_loss < best_loss - 1e-5:
        best_loss = val_loss
        best_epoch = epoch + 1
        best_weights = deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f}"
        )

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch + 1}")
        break

model.load_state_dict(best_weights)
model.eval()

print("Best epoch:", best_epoch)
print("Best validation loss:", best_loss)
# %% Training and validation loss curves
plt.figure(figsize=(9, 4))

plt.plot(
    history["train_loss"],
    label="Training loss",
)

plt.plot(
    history["val_loss"],
    label="Future-unseen validation loss",
)

plt.axvline(
    best_epoch - 1,
    color="black",
    linestyle="--",
    label=f"Best epoch: {best_epoch}",
)

plt.xlabel("Epoch (zero-indexed on plot)")
plt.ylabel("Binary Cross-Entropy")
plt.title("Outlet Embedding MLP — Learning Curves")
plt.legend()
plt.tight_layout()
plt.show()
# %% Predict and evaluate
def predict_outlet_model(model, data):

    model.eval()

    X_outlet = torch.tensor(
        data["outlet"],
        device=device,
    )

    X_context = torch.tensor(
        data["context"],
        device=device,
    )

    with torch.no_grad():
        logits = model(X_outlet, X_context)
        probabilities = torch.sigmoid(logits)

    return probabilities.cpu().numpy().ravel()


results = []
prediction_tables = {}

for name, data in processed.items():

    y_prob = predict_outlet_model(model, data)

    metrics = evaluate_predictions(
        data["y"],
        y_prob,
    )

    results.append({
        "model": "outlet_embedding_mlp",
        "experiment": EXPERIMENT_NAME,
        "dataset": name,
        "best_epoch": best_epoch,
        "n_parameters": n_parameters,
        **metrics,
    })

    prediction_tables[name] = pd.DataFrame({
        "outlet": datasets[name]["outlet"].to_numpy(),
        "product": datasets[name]["product"].to_numpy(),
        "y_true": data["y"],
        "y_prob": y_prob,
    })

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))
# %% Probability calibration — Outlet Embedding MLP
for name in ["future_known", "future_unseen"]:

    predictions = prediction_tables[name]

    plot_calibration(
        predictions["y_true"],
        predictions["y_prob"],
        title=f"Outlet Embedding MLP — {name}"
    )
# %% Extract outlet embeddings
def extract_outlet_embeddings(model, X_outlet):

    model.eval()

    X_tensor = torch.tensor(
        X_outlet,
        device=device,
    )

    with torch.no_grad():
        embeddings = model.encode_outlet(
            X_tensor
        )

    return embeddings.cpu().numpy()


train_embeddings = extract_outlet_embeddings(
    model,
    processed["train"]["outlet"],
)

embedding_cols = [
    f"embedding_{i}"
    for i in range(train_embeddings.shape[1])
]

embedding_df = pd.DataFrame(
    train_embeddings,
    columns=embedding_cols,
)

embedding_df["outlet"] = train_df["outlet"].to_numpy()
embedding_df["outlet_type"] = (
    train_df["outlet_type"].to_numpy()
)

# Aggregate repeated observations into one point per outlet
outlet_embeddings_df = (
    embedding_df
    .groupby("outlet", as_index=False)
    .agg({
        **{col: "mean" for col in embedding_cols},
        "outlet_type": "first",
    })
)

print(outlet_embeddings_df.head())
print("Outlets represented:", len(outlet_embeddings_df))
# %% PCA visualization of outlet embeddings
X_embeddings = outlet_embeddings_df[
    embedding_cols
].to_numpy()

pca = PCA(n_components=2)

coordinates = pca.fit_transform(X_embeddings)

plot_df = outlet_embeddings_df.copy()
plot_df["PC1"] = coordinates[:, 0]
plot_df["PC2"] = coordinates[:, 1]

plt.figure(figsize=(9, 6))

for outlet_type, group in plot_df.groupby(
    "outlet_type",
    dropna=False,
):
    plt.scatter(
        group["PC1"],
        group["PC2"],
        label=str(outlet_type),
        alpha=0.6,
        s=25,
    )

plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.title("Learned Outlet Representations — PCA")
plt.legend()
plt.tight_layout()
plt.show()

print(
    "Variance explained by two components:",
    pca.explained_variance_ratio_.sum(),
)
# %% Final outlet embedding model evaluation on test set
test_df = pd.read_csv(
    DATA_DIR / "test_future_unseen.csv"
)

# Prepare the two feature groups
X_test_outlet = prepare_model_features(
    test_df,
    OUTLET_NUMERIC,
    OUTLET_CATEGORICAL,
)

X_test_context = prepare_model_features(
    test_df,
    CONTEXT_NUMERIC,
    CONTEXT_CATEGORICAL,
)

# Transform using the fitted training preprocessors
test_data = {
    "outlet": np.asarray(
        outlet_preprocessor.transform(X_test_outlet),
        dtype=np.float32,
    ),
    "context": np.asarray(
        context_preprocessor.transform(X_test_context),
        dtype=np.float32,
    ),
    "y": test_df[TARGET].to_numpy(dtype=np.float32),
}

# Uses the existing prediction function from Cell 8
y_prob_test = predict_outlet_model(
    model,
    test_data,
)

test_metrics = evaluate_predictions(
    test_data["y"],
    y_prob_test,
)

test_results_df = pd.DataFrame([{
    "model": "outlet_embedding_mlp",
    "experiment": EXPERIMENT_NAME,
    "dataset": "test_future_unseen",
    "best_epoch": best_epoch,
    "n_parameters": n_parameters,
    **test_metrics,
}])

print(test_results_df.to_string(index=False))
# %%
# Inspect the embedding space
from sklearn.metrics.pairwise import cosine_similarity

# Keep one representation per outlet
embedding_matrix = outlet_embeddings_df[
    embedding_cols
].to_numpy()

similarity_matrix = cosine_similarity(
    embedding_matrix
)

outlet_ids = outlet_embeddings_df["outlet"].to_numpy()


def find_similar_outlets(outlet_id, top_k=5):

    matches = np.flatnonzero(outlet_ids == outlet_id)

    if len(matches) == 0:
        raise ValueError(f"Outlet not found: {outlet_id}")

    idx = matches[0]

    scores = similarity_matrix[idx].copy()

    # Exclude the selected outlet itself
    scores[idx] = -np.inf

    closest_indices = np.argsort(scores)[::-1][:top_k]

    return pd.DataFrame({
        "outlet": outlet_ids[closest_indices],
        "cosine_similarity": scores[closest_indices],
    })


# Example: choose an outlet from the training data
REFERENCE_OUTLET = outlet_ids[0]

similar_outlets = find_similar_outlets(
    REFERENCE_OUTLET,
    top_k=5,
)

print(f"Reference outlet: {REFERENCE_OUTLET}")
print(similar_outlets.to_string(index=False))

def representative_value(series):

    non_missing = series.dropna()

    if non_missing.empty:
        return np.nan

    if pd.api.types.is_numeric_dtype(series):
        return non_missing.median()

    return non_missing.mode().iloc[0]


outlet_profiles = (
    train_df
    .groupby("outlet")[OUTLET_FEATURES]
    .agg(representative_value)
)


def compare_similar_outlets(outlet_id, top_k=5):

    neighbors = find_similar_outlets(
        outlet_id,
        top_k=top_k,
    )

    selected_ids = [
        outlet_id,
        *neighbors["outlet"].tolist(),
    ]

    # Transpose: features become rows, outlets columns
    comparison = outlet_profiles.loc[
        selected_ids,
        OUTLET_FEATURES,
    ].T

    # Add similarity scores as the first row
    similarity_values = {
        outlet_id: 1.0,
        **dict(zip(
            neighbors["outlet"],
            neighbors["cosine_similarity"],
        )),
    }

    comparison.loc[
        "cosine_similarity"
    ] = pd.Series(similarity_values)

    comparison = comparison.reindex(
        ["cosine_similarity", *OUTLET_FEATURES]
    )

    return comparison


comparison_df = compare_similar_outlets(
    REFERENCE_OUTLET,
    top_k=5,
)

print(
    comparison_df.to_string()
)
# %%
