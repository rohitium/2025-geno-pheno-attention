#!/usr/bin/env python3
"""
Standalone Modal script to predict growth at arbitrary temperatures.
Mounts local model files and uses existing analysis package.
"""

import modal

app = modal.App("temperature-prediction-standalone")

# Create image with required dependencies and model file
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.2.2",
        "numpy==1.26.4",
        "scipy==1.16.1",
        "lightning==2.5.1",
        "attrs==25.3.0",
    )
    .add_local_python_source("analysis")  # Add the analysis package
    .add_local_file(
        "models/transformer/xformer_rep_00/lightning_logs/version_0/checkpoints/best-epoch=021-val_loss=0.3811.ckpt",
        "/checkpoint.ckpt"
    )
)

@app.function(
    image=image,
    gpu="A10G:1",
    timeout=600,
)
def run_temperature_prediction():
    from pathlib import Path

    import numpy as np
    import torch
    from analysis.base import ModelConfig, TrainConfig
    from analysis.dataset import phenotype_names
    from analysis.transformer import Transformer
    from scipy.interpolate import interp1d

    print("🔬 Loading pre-trained transformer...")

    # Model checkpoint is now available at /checkpoint.ckpt
    checkpoint = Path("/checkpoint.ckpt")

    model_config = ModelConfig(
        model_type="transformer",
        seq_length=1164,
        embedding_dim=256,
        num_layers=3,
        nhead=4,
        dim_feedforward=1024
    )
    train_config = TrainConfig(phenotypes=phenotype_names)

    transformer = Transformer.load_from_checkpoint(
        str(checkpoint),
        model_config=model_config,
        train_config=train_config
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

    # Single forward pass
    with torch.no_grad():
        predictions = transformer(unseen_genotypes)

    # Extract temperature predictions
    temp_values = [23, 25, 27, 30, 33, 35, 37]
    temp_indices = [phenotype_names.index(f"{t}C") for t in temp_values]
    temp_predictions = predictions[:, temp_indices].cpu()

    print("🌡️  Predicting growth at 28.5°C using cubic spline...")

    arbitrary_temp = 28.5
    results = []
    for i in range(batch_size):
        interp_func = interp1d(temp_values, temp_predictions[i].numpy(),
                             kind='cubic', fill_value='extrapolate')
        results.append(interp_func(arbitrary_temp))

    results = np.array(results)
    print(f"Growth at {arbitrary_temp}°C: mean={results.mean():.3f}, std={results.std():.3f}")

    print("✅ Temperature prediction completed successfully on GPU!")
    return {"batch_size": batch_size, "device": str(device), "temperature": arbitrary_temp}

@app.local_entrypoint()
def main():
    print("🚀 Running temperature prediction on Modal...")
    result = run_temperature_prediction.remote()
    print(f"✅ Modal job completed: {result}")

if __name__ == "__main__":
    main()
