# %% Define neural network architecture
import torch
from torch import nn

class AvailabilityMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim1=32,
        hidden_dim2=16,
        dropout=0.1,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x):
        return self.network(x)


class OutletEmbeddingMLP(nn.Module):

    def __init__(
        self,
        outlet_input_dim,
        context_input_dim,
        embedding_dim=8,
        encoder_hidden_dim=16,
        prediction_hidden_dim=16,
        dropout=0.1,
    ):
        super().__init__()

        # Learn a compact outlet representation
        self.outlet_encoder = nn.Sequential(
            nn.Linear(outlet_input_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, embedding_dim),
        )

        # Combine the outlet representation
        # with product and calendar features
        self.prediction_head = nn.Sequential(
            nn.Linear(
                embedding_dim + context_input_dim,
                prediction_hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prediction_hidden_dim, 1),
        )

    def encode_outlet(self, X_outlet):
        return self.outlet_encoder(X_outlet)

    def forward(self, X_outlet, X_context):

        embedding = self.encode_outlet(X_outlet)

        combined = torch.cat(
            [embedding, X_context],
            dim=1,
        )

        logits = self.prediction_head(combined)

        return logits