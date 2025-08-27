#!/usr/bin/env python3
"""
Minimal script to train rijal_et_al model for 30C phenotype and generate predictions.
This script trains from scratch without using caching, using Modal for distributed training.
"""

from pathlib import Path

import pandas as pd
import torch
from analysis.base import ModelConfig, TrainConfig
from analysis.dataset import create_dataloaders
from analysis.rijal_et_al import RijalEtAl
from analysis.train import run_training


def main():
    """Train rijal_et_al model for 30C phenotype and generate predictions."""
    # Dataset directory (using existing downloaded data)
    dataset_dir = Path("datasets/")
    if not dataset_dir.exists():
        print(f"Error: Dataset directory {dataset_dir} not found.")
        print("Please ensure datasets are downloaded first.")
        return

    # Model configuration - matching original Rijal et al. architecture
    model_config = ModelConfig(
        model_type="rijal_et_al",
        seq_length=1164,
        embedding_dim=13,
        num_layers=3,
    )

    # Training configuration for 30C phenotype
    train_config = TrainConfig(
        data_dir=dataset_dir,
        save_dir=Path("models/30C_prediction"),
        name_prefix="30C_no_cache",
        phenotypes=["30C"],  # Single phenotype
        optimizer="adam",
        batch_size=64,
        learning_rate=0.001,
        lr_schedule=False,
        weight_decay=0.0,
        max_epochs=200,
        gradient_clip_val=0,
        use_cache=False,  # Force training without cache
        use_modal=True,  # Use Modal for GPU training
        modal_detach=True,  # Allow detaching from Modal job
        seed=42,  # Fixed seed for reproducibility
    )

    print("Starting training for 30C phenotype using Modal...")
    print(f"Model: {model_config.model_type}")
    print(f"Phenotype: {train_config.phenotypes[0]}")
    print(f"Cache disabled: {not train_config.use_cache}")
    print(f"Using Modal: {train_config.use_modal}")

    # Train the model
    model_dir = run_training(model_config, train_config)
    print(f"Training completed. Model saved to: {model_dir}")

    # Load the best checkpoint for predictions
    checkpoint_path = model_dir / "checkpoints"
    best_checkpoint = list(checkpoint_path.glob("best-*.ckpt"))[0]
    print(f"Loading best checkpoint: {best_checkpoint}")

    # Load model from checkpoint
    model = RijalEtAl.load_from_checkpoint(best_checkpoint)
    model.eval()

    # Create test dataloader
    _, _, test_loader = create_dataloaders(
        data_dir=train_config.data_dir,
        phenotypes=train_config.phenotypes,
        batch_size=train_config.batch_size,
        num_workers=1,
        synthetic=train_config.synthetic_data,
    )

    # Generate predictions on test set
    predictions = []
    targets = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for batch_idx, (genotypes, phenotypes) in enumerate(test_loader):
            # Strip NaN values (matching model's _strip_nan method)
            mask = ~torch.isnan(phenotypes).any(dim=1)
            if mask.sum() == 0:
                continue  # Skip all-NaN batch

            clean_genotypes = genotypes[mask]
            clean_phenotypes = phenotypes[mask]

            # Get predictions
            pred = model(clean_genotypes)

            predictions.extend(pred.cpu().numpy().flatten())
            targets.extend(clean_phenotypes.cpu().numpy().flatten())

            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(test_loader)}")

    # Create results DataFrame
    results_df = pd.DataFrame({"true_30C": targets, "predicted_30C": predictions})

    # Calculate R² using numpy correlation coefficient
    import numpy as np

    correlation_matrix = np.corrcoef(targets, predictions)
    r2 = correlation_matrix[0, 1] ** 2

    print("\nResults Summary:")
    print(f"Number of test samples: {len(targets)}")
    print(f"Test R² score: {r2:.4f}")

    # Save predictions
    output_file = "30C_predictions.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to: {output_file}")

    # Display first few predictions
    print("\nFirst 10 predictions:")
    print(results_df.head(10))


if __name__ == "__main__":
    main()
