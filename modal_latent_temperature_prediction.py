#!/usr/bin/env python3
"""
Modal script for latent-based temperature prediction using transformer embeddings.
"""

import modal

app = modal.App("latent-temperature-prediction")

# Create image with required dependencies and model file
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.2.2",
        "numpy==1.26.4",
        "lightning==2.5.1",
        "attrs==25.3.0",
    )
    .add_local_python_source("analysis")
    .add_local_file(
        "models/transformer/xformer_rep_00/lightning_logs/version_0/checkpoints/best-epoch=021-val_loss=0.3811.ckpt",
        "/checkpoint.ckpt",
    )
)


@app.function(
    image=image,
    gpu="A10G:1",
    timeout=600,
)
def run_latent_temperature_prediction():
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

    print("🔬 Loading pre-trained transformer...")

    # Model checkpoint is available at /checkpoint.ckpt
    checkpoint = Path("/checkpoint.ckpt")

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
        str(checkpoint), model_config=model_config, train_config=train_config
    )
    transformer.eval()

    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = transformer.to(device)
    print(f"✅ Model loaded on device: {device}")

    # Generate unseen genetic data
    batch_size = 1000
    unseen_genotypes = torch.randn(batch_size, 1164, device=device)
    print(f"🧬 Generated {batch_size} random genotypes on {device}")

    # Extract latent embeddings from transformer encoder
    with torch.no_grad():
        # Get embeddings by replicating the forward pass up to encoder output
        locus_embeddings = transformer.model.locus_embeddings(transformer.model.locus_indices)
        x = unseen_genotypes.unsqueeze(-1) * locus_embeddings
        encoder_output = transformer.model.transformer_encoder(src=x)
        embeddings = encoder_output.mean(dim=1)  # Pool like the original model

        full_predictions = transformer(unseen_genotypes)

    # Prepare training data for temperature predictor
    temp_values = [23, 25, 27, 30, 33, 35, 37]
    temp_indices = [phenotype_names.index(f"{t}C") for t in temp_values]
    temp_predictions = full_predictions[:, temp_indices]

    # Create training dataset: (embedding, temperature) -> growth
    train_embeddings = []
    train_temperatures = []
    train_targets = []

    for i, temp in enumerate(temp_values):
        train_embeddings.append(embeddings)
        train_temperatures.append(torch.full((batch_size, 1), temp, device=device))
        train_targets.append(temp_predictions[:, i : i + 1])

    # Combine all training data
    train_embeddings = torch.cat(train_embeddings, dim=0)
    train_temperatures = torch.cat(train_temperatures, dim=0)
    train_targets = torch.cat(train_targets, dim=0)

    # Train temperature predictor
    temp_predictor = TemperaturePredictor(embedding_dim=256).to(device)
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

    # Predict at arbitrary temperature using latent representations
    arbitrary_temp = 28.5
    temp_predictor.eval()

    with torch.no_grad():
        temp_tensor = torch.full((batch_size, 1), arbitrary_temp, device=device)
        results = temp_predictor(embeddings, temp_tensor).squeeze().cpu().numpy()

    print(
        f"Growth at {arbitrary_temp}°C (latent method): "
        f"mean={np.mean(results):.3f}, std={np.std(results):.3f}"
    )

    print("✅ Latent temperature prediction completed successfully on GPU!")
    return {
        "batch_size": batch_size,
        "device": str(device),
        "temperature": arbitrary_temp,
        "method": "latent",
    }


@app.local_entrypoint()
def main():
    print("🚀 Running latent temperature prediction on Modal...")
    result = run_latent_temperature_prediction.remote()
    print(f"✅ Modal job completed: {result}")


if __name__ == "__main__":
    main()
