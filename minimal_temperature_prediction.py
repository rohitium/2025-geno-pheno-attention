#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import torch
from analysis.base import ModelConfig, TrainConfig
from analysis.dataset import phenotype_names
from analysis.transformer import Transformer
from scipy.interpolate import interp1d

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

# Generate random genetic data and predict
unseen_genotypes = torch.randn(1000, 1164)  # B=1000, L=1164
with torch.no_grad():
    predictions = transformer(unseen_genotypes)  # B=1000, P=18

# Extract temperature predictions and interpolate
temp_values = [23, 25, 27, 30, 33, 35, 37]  # 7 temperatures
temp_indices = [phenotype_names.index(f"{t}C") for t in temp_values]  # 7 indices
temp_predictions = predictions[:, temp_indices]  # B=1000, P=7

# Predict at arbitrary temperature (e.g., 28.5°C)
arbitrary_temp = 28.5
results = [
    interp1d(temp_values, temp_predictions[i].numpy(), kind="cubic", fill_value="extrapolate")(
        arbitrary_temp
    )
    for i in range(1000)
]  # B=1000, P=1

print(f"Growth at {arbitrary_temp}°C: mean={np.mean(results):.3f}, std={np.std(results):.3f}")
