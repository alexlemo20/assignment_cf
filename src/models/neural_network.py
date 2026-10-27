# %% Imports and setup
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tqdm.auto import tqdm

from copy import deepcopy
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


from src.models.utils import (
    evaluate_predictions,
    plot_calibration,
    ORDINAL_COLS,
    NUMERIC_COLS,
    BINARY_COLS,
    CALENDAR_COLS,
    NOMINAL_COLS,
    prepare_model_features,
    build_mlp_preprocessor,
    add_prediction_errors,
    summarize_prediction_errors,
)

from src.models.MLP import AvailabilityMLP


DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "reports" / "model_results"

EXPERIMENT_NAME = "mlp_default"
EXPERIMENT_DIR = RESULTS_DIR / EXPERIMENT_NAME
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "availability"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("PyTorch:", torch.__version__)
print("Device:", device)
# %% Load prepared datasets
train_df = pd.read_csv(DATA_DIR / "train.csv")

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
# %% Prepare input features
def prepare_features(dataset):
    return prepare_model_features(
        dataset, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
    )


X_train = prepare_features(train_df)
X_val_known = prepare_features(val_known_df)
X_val_unseen = prepare_features(val_unseen_df)

y_train = train_df[TARGET].to_numpy(dtype=np.float32)
y_val_known = val_known_df[TARGET].to_numpy(dtype=np.float32)
y_val_unseen = val_unseen_df[TARGET].to_numpy(dtype=np.float32)

print(X_train.dtypes)
# %% Build preprocessing pipeline
preprocessor = build_mlp_preprocessor(
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)
# %%
# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

X_val_known_processed = preprocessor.transform(X_val_known)
X_val_unseen_processed = preprocessor.transform(X_val_unseen)

# Convert to float32 for PyTorch
X_train_processed = np.asarray(
    X_train_processed,
    dtype=np.float32
)

X_val_known_processed = np.asarray(
    X_val_known_processed,
    dtype=np.float32
)

X_val_unseen_processed = np.asarray(
    X_val_unseen_processed,
    dtype=np.float32
)

print("Encoded input dimensions:", X_train_processed.shape[1])
print("Training matrix:", X_train_processed.shape)
# %% Build tensors and DataLoader

X_train_tensor = torch.tensor(
    X_train_processed,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
).reshape(-1, 1)

X_val_tensor = torch.tensor(
    X_val_unseen_processed,
    dtype=torch.float32,
    device=device,
)

y_val_tensor = torch.tensor(
    y_val_unseen,
    dtype=torch.float32,
    device=device,
).reshape(-1, 1)

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)

print("Training batches:", len(train_loader))

# Initialize network
INPUT_DIM = X_train_processed.shape[1]

model = AvailabilityMLP(
    input_dim=INPUT_DIM,
    hidden_dim1=16,
    hidden_dim2=8,
    dropout=0.1,
).to(device)

n_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(model)
print("Number of trainable parameters:", n_parameters)
# %% Training configuration

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-3,
)

MAX_EPOCHS = 200
PATIENCE = 15

best_val_loss = float("inf")
best_epoch = 0
best_weights = None
epochs_without_improvement = 0

training_history = {
    "train_loss": [],
    "val_loss": [],
}

# %% Train neural network
for epoch in tqdm(
    range(MAX_EPOCHS),
    leave=False
    ):

    # -----------------------
    # Training
    # -----------------------
    model.train()

    total_train_loss = 0.0

    for X_batch, y_batch in tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{MAX_EPOCHS}",
        leave=False
    ):

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)

        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_train_loss += (
            loss.item() * len(X_batch)
        )

    avg_train_loss = total_train_loss / len(train_dataset)

    # -----------------------
    # Validation
    # -----------------------
    model.eval()

    with torch.no_grad():

        val_logits = model(X_val_tensor)

        val_loss = criterion(
            val_logits,
            y_val_tensor
        ).item()

    training_history["train_loss"].append(avg_train_loss)
    training_history["val_loss"].append(val_loss)

    # -----------------------
    # Early stopping
    # -----------------------
    if val_loss < best_val_loss - 1e-5:

        best_val_loss = val_loss
        best_epoch = epoch + 1
        best_weights = deepcopy(model.state_dict())
        epochs_without_improvement = 0

    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch + 1}")
        break

# Restore best validation model
model.load_state_dict(best_weights)
model.eval()

print("Best epoch:", best_epoch)
print("Best validation loss:", best_val_loss)
# %% Training curves
epochs = np.arange(
    1,
    len(training_history["train_loss"]) + 1
)

plt.figure(figsize=(10, 5))

plt.plot(
    epochs,
    training_history["train_loss"],
    label="Training Loss"
)

plt.plot(
    epochs,
    training_history["val_loss"],
    label="Unseen Validation Loss"
)

plt.axvline(
    best_epoch,
    color="black",
    linestyle="--",
    label=f"Best epoch: {best_epoch}"
)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("MLP — Training and Validation Loss")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %% Predict probabilities
def predict_proba(model, X, device):

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():

        logits = model(X_tensor)

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy().ravel()

    return probabilities

datasets = {
    "train": (
        X_train_processed,
        y_train,
        train_df
    ),
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

for name, (X, y, original_df) in datasets.items():

    y_prob = predict_proba(model, X, device)

    metrics = evaluate_predictions(
        y,
        y_prob
    )

    results.append({
        "model": "mlp",
        "experiment": EXPERIMENT_NAME,
        "dataset": name,
        "best_epoch": best_epoch,
        "n_parameters": n_parameters,
        **metrics
    })

    prediction_tables[name] = pd.DataFrame({
        "outlet": original_df["outlet"].to_numpy(),
        "product": original_df["product"].to_numpy(),
        "survey_date": original_df["survey_date"].to_numpy(),
        "y_true": y,
        "y_prob": y_prob,
    })

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))
# %% Calibration plot
pred_df = prediction_tables["future_unseen"]

plot_calibration(
    pred_df["y_true"],
    pred_df["y_prob"],
    title=f"MLP — future_unseen"
)

pred_df = prediction_tables["future_known"]

plot_calibration(
    pred_df["y_true"],
    pred_df["y_prob"],
    title=f"MLP — future_known"
)
# %% Error analysis by product and outlet type

error_df = val_unseen_df.copy()

error_df["y_true"] = prediction_tables[
    "future_unseen"
]["y_true"].to_numpy()

error_df["y_prob"] = prediction_tables[
    "future_unseen"
]["y_prob"].to_numpy()

error_df = add_prediction_errors(error_df)

for feature in ["product", "outlet_type"]:

    summary = summarize_prediction_errors(error_df, feature)

    print(f"\nError analysis by {feature}")
    print(summary.to_string())

# %% Final MLP evaluation on test set
model.eval()

test_df = pd.read_csv(
    DATA_DIR / "test_future_unseen.csv"
)

X_test = prepare_features(test_df)
y_test = test_df[TARGET].to_numpy(dtype=np.float32)

# Apply the fitted preprocessing from training
X_test_processed = preprocessor.transform(X_test)

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float32
)

# Predict probabilities using the trained MLP
y_prob_test = predict_proba(
    model,
    X_test_processed,
    device
)

test_metrics = evaluate_predictions(
    y_test,
    y_prob_test
)

test_results_df = pd.DataFrame([{
    "model": "mlp",
    "experiment": EXPERIMENT_NAME,
    "dataset": "test_future_unseen",
    "best_epoch": best_epoch,
    "n_parameters": n_parameters,
    **test_metrics,
}])

print(test_results_df.to_string(index=False))
# %%
