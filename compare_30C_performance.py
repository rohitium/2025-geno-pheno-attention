#!/usr/bin/env python3
"""
Compare test R² performance for 30C phenotype across different model architectures:
1. Original rijal_et_al (fig3)
2. Our recent rijal_et_al run (30C_prediction)
3. Modified rijal_et_al (canonical)
4. Transformer
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_30C_r2_from_metrics(metrics_path: Path) -> float:
    """Extract the test R² for 30C phenotype from metrics.csv file."""
    if not metrics_path.exists():
        return np.nan

    df = pd.read_csv(metrics_path)

    # Look for 30C-specific R² first
    test_r2_30C = df[df["metric"] == "test_r2_30C"]
    if not test_r2_30C.empty:
        return float(test_r2_30C["value"].iloc[0])

    # Fallback to general test_r2 (for single-phenotype models)
    test_r2_general = df[df["metric"] == "test_r2"]
    if not test_r2_general.empty:
        return float(test_r2_general["value"].iloc[0])

    return np.nan


def main():
    """Compare 30C test R² performance across different model architectures."""

    # Define model paths and their display names
    models = {
        "Original rijal_et_al\n(fig3)": (
            "models/fig3/fig3_30C/lightning_logs/version_0/metrics.csv"
        ),
        "Our rijal_et_al\n(30C_prediction)": (
            "models/30C_prediction/30C_no_cache/lightning_logs/version_0/metrics.csv"
        ),
        "Modified rijal_et_al\n(canonical)": (
            "models/canonical/std_d128_rep_00/lightning_logs/version_0/metrics.csv"
        ),
        "Transformer": ("models/transformer/xformer_rep_00/lightning_logs/version_0/metrics.csv"),
    }

    # Extract R² values
    results = {}
    for model_name, metrics_path in models.items():
        metrics_file = Path(metrics_path)
        r2_value = extract_30C_r2_from_metrics(metrics_file)
        results[model_name] = r2_value

        # Print status
        if not metrics_file.exists():
            print(f"⚠️  {model_name}: metrics.csv not found at {metrics_path}")
        elif np.isnan(r2_value):
            print(f"⚠️  {model_name}: Could not extract 30C R² from metrics")
        else:
            print(f"✓  {model_name}: R² = {r2_value:.4f}")

    # Filter out NaN values for plotting
    valid_results = {k: v for k, v in results.items() if not np.isnan(v)}

    if not valid_results:
        print("❌ No valid R² values found. Please check the model paths.")
        return

    # Create comparison plot
    model_names = list(valid_results.keys())
    r2_values = list(valid_results.values())

    plt.figure(figsize=(12, 8))

    # Create bar plot with different colors
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
    bars = plt.bar(
        model_names, r2_values, color=colors[: len(model_names)], alpha=0.8, edgecolor="black"
    )

    # Add value labels on bars
    for bar, value in zip(bars, r2_values, strict=False):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    # Customize plot
    plt.ylabel("Test R² Score", fontsize=14, fontweight="bold")
    plt.title(
        "30C Phenotype Test R² Comparison\nAcross Different Model Architectures",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.ylim(0, max(r2_values) * 1.15)  # Add some headroom for labels

    # Add grid for better readability
    plt.grid(axis="y", alpha=0.3, linestyle="--")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0, ha="center", fontsize=11)
    plt.yticks(fontsize=11)

    # Add horizontal line at 0.6 for reference
    plt.axhline(y=0.6, color="red", linestyle="--", alpha=0.7, linewidth=2)
    plt.text(
        0.02,
        0.605,
        "R² = 0.6",
        transform=plt.gca().get_yaxis_transform(),
        color="red",
        fontweight="bold",
    )

    plt.tight_layout()

    # Save plot
    output_file = "30C_performance_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n📊 Plot saved to: {output_file}")

    # Show plot
    plt.show()

    # Create summary table
    print("\n" + "=" * 60)
    print("30C PHENOTYPE TEST R² COMPARISON SUMMARY")
    print("=" * 60)

    # Sort by R² value (descending)
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1], reverse=True)

    for i, (model, r2) in enumerate(sorted_results):
        rank = i + 1
        print(f"{rank}. {model.replace(chr(10), ' '):<35} R² = {r2:.4f}")

    print("=" * 60)

    # Calculate improvements
    if len(sorted_results) >= 2:
        best_model, best_r2 = sorted_results[0]
        baseline_model, baseline_r2 = sorted_results[-1]
        improvement = ((best_r2 - baseline_r2) / baseline_r2) * 100
        print(f"\nBest model improvement over baseline: {improvement:.1f}%")
        print(f"Best: {best_model.replace(chr(10), ' ')} (R² = {best_r2:.4f})")
        print(f"Baseline: {baseline_model.replace(chr(10), ' ')} (R² = {baseline_r2:.4f})")


if __name__ == "__main__":
    main()
