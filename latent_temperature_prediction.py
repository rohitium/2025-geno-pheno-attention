#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from analysis.base import ModelConfig, TrainConfig
from analysis.dataset import phenotype_names
from analysis.transformer import Transformer


# Simple temperature predictor network
class TemperaturePredictor(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim + 1, 128),  # +1 for temperature input
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Single output: growth prediction
        )

    def forward(self, embeddings, temperature):
        # embeddings: [B, embedding_dim], temperature: [B, 1]
        combined = torch.cat([embeddings, temperature], dim=1)
        return self.predictor(combined)


# Load transformer
checkpoint = list(
    Path("models/transformer/xformer_rep_00/lightning_logs/version_0/checkpoints").glob(
        "best-*.ckpt"
    )
)[0]
model_config = ModelConfig(
    model_type="transformer",
    seq_length=1164,
    embedding_dim=256,
    num_layers=3,
    nhead=4,
    dim_feedforward=1024,
)
train_config = TrainConfig(phenotypes=phenotype_names)
transformer = Transformer.load_from_checkpoint(
    checkpoint, model_config=model_config, train_config=train_config
)
transformer.eval()

# Generate random genetic data
unseen_genotypes = torch.randn(1000, 1164)  # B=1000, L=1164

# Extract latent embeddings from transformer encoder
with torch.no_grad():
    # Get embeddings from transformer encoder (before final prediction head)
    embeddings = transformer.transformer_encoder(unseen_genotypes)  # B=1000, embedding_dim=256
    # Get ground truth predictions for training the temperature predictor
    full_predictions = transformer(unseen_genotypes)  # B=1000, P=18

# Prepare training data for temperature predictor
temp_values = [23, 25, 27, 30, 33, 35, 37]  # 7 temperatures
temp_indices = [phenotype_names.index(f"{t}C") for t in temp_values]
temp_predictions = full_predictions[:, temp_indices]  # B=1000, P=7

# Create training dataset: (embedding, temperature) -> growth
train_embeddings = []
train_temperatures = []
train_targets = []

for i, temp in enumerate(temp_values):
    train_embeddings.append(embeddings)  # B=1000, embedding_dim
    train_temperatures.append(torch.full((1000, 1), temp))  # B=1000, 1
    train_targets.append(temp_predictions[:, i : i + 1])  # B=1000, 1

# Combine all training data
train_embeddings = torch.cat(train_embeddings, dim=0)  # B=7000, embedding_dim
train_temperatures = torch.cat(train_temperatures, dim=0)  # B=7000, 1
train_targets = torch.cat(train_targets, dim=0)  # B=7000, 1

# Train temperature predictor
temp_predictor = TemperaturePredictor(embedding_dim=256)
optimizer = torch.optim.Adam(temp_predictor.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("🧠 Training temperature predictor on latent embeddings...")
for epoch in range(100):
    optimizer.zero_grad()
    predicted = temp_predictor(train_embeddings, train_temperatures)
    loss = criterion(predicted, train_targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch + 1}/100, Loss: {loss.item():.4f}")

# Now predict at arbitrary temperature using latent representations
arbitrary_temp = 28.5
temp_predictor.eval()

with torch.no_grad():
    temp_tensor = torch.full((1000, 1), arbitrary_temp)
    results = temp_predictor(embeddings, temp_tensor).squeeze().numpy()

print(
    f"Growth at {arbitrary_temp}°C (latent method): "
    f"mean={np.mean(results):.3f}, std={np.std(results):.3f}"
)
