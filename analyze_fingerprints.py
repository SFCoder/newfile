"""
SwiGLU Fingerprint Analysis
============================
Compares activation fingerprints between two models (e.g., teacher vs student)
and quantifies how distinguishable they are.

Usage:
    python analyze_fingerprints.py \
        --model-a ./fingerprint_data/70b \
        --model-b ./fingerprint_data/8b \
        --output ./analysis_results

Produces:
    - Statistical separation metrics
    - Per-layer comparison plots
    - Trajectory analysis
    - Classification accuracy (can a simple classifier tell them apart?)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold


def load_fingerprints(model_dir: str) -> tuple[dict, list]:
    """Load metadata and fingerprint data from extraction output."""
    model_path = Path(model_dir)

    with open(model_path / "metadata.json") as f:
        metadata = json.load(f)

    with open(model_path / "fingerprints.json") as f:
        fingerprints = json.load(f)

    print(f"Loaded {len(fingerprints)} fingerprints from {metadata['model_name']} "
          f"({metadata['tag']})")
    print(f"  Architecture: {metadata['num_layers']}L / {metadata['hidden_dim']}d / "
          f"{metadata['num_attention_heads']}H")

    return metadata, fingerprints


# ---------------------------------------------------------------------------
# 1. Per-layer distribution comparison
# ---------------------------------------------------------------------------

def compare_per_layer_stats(
    meta_a, fp_a, meta_b, fp_b, output_dir: Path
):
    """
    For each statistic, compare distributions across the two models layer-by-layer.
    """
    print("\n" + "="*70)
    print("PER-LAYER STATISTICAL COMPARISON")
    print("="*70)

    stats_to_compare = [
        "sparsity_ratio",
        "gini_mean",
        "l1_l2_ratio",
        "mag_mean",
        "mag_std",
        "mag_skew",
        "mag_kurtosis",
        "energy_concentration_top1pct",
    ]

    # Determine layer counts
    layers_a = sorted(int(k) for k in fp_a[0]["layers"].keys())
    layers_b = sorted(int(k) for k in fp_b[0]["layers"].keys())

    results = {}

    for stat_name in stats_to_compare:
        print(f"\n--- {stat_name} ---")

        # Collect per-layer values for each model
        values_a = {l: [] for l in layers_a}
        values_b = {l: [] for l in layers_b}

        for fp in fp_a:
            for l in layers_a:
                val = fp["layers"].get(str(l), {}).get(stat_name)
                if val is not None:
                    values_a[l].append(val)

        for fp in fp_b:
            for l in layers_b:
                val = fp["layers"].get(str(l), {}).get(stat_name)
                if val is not None:
                    values_b[l].append(val)

        # Compare at shared layer indices (normalized by depth)
        # For models with different layer counts, map by relative depth
        results[stat_name] = {
            "model_a": {
                "per_layer_mean": {l: np.mean(values_a[l]) for l in layers_a if values_a[l]},
                "per_layer_std": {l: np.std(values_a[l]) for l in layers_a if values_a[l]},
            },
            "model_b": {
                "per_layer_mean": {l: np.mean(values_b[l]) for l in layers_b if values_b[l]},
                "per_layer_std": {l: np.std(values_b[l]) for l in layers_b if values_b[l]},
            },
        }

        # Global comparison (all layers averaged)
        all_a = [v for l in layers_a for v in values_a[l]]
        all_b = [v for l in layers_b for v in values_b[l]]

        if all_a and all_b:
            cohens_d = (np.mean(all_a) - np.mean(all_b)) / np.sqrt(
                (np.std(all_a)**2 + np.std(all_b)**2) / 2
            )
            ks_stat, ks_pval = stats.ks_2samp(all_a, all_b)

            print(f"  Model A mean: {np.mean(all_a):.4f} ± {np.std(all_a):.4f}")
            print(f"  Model B mean: {np.mean(all_b):.4f} ± {np.std(all_b):.4f}")
            print(f"  Cohen's d:    {cohens_d:.3f}")
            print(f"  KS statistic: {ks_stat:.4f}  (p={ks_pval:.2e})")

            results[stat_name]["cohens_d"] = cohens_d
            results[stat_name]["ks_statistic"] = ks_stat
            results[stat_name]["ks_pvalue"] = ks_pval

        # --- Plot ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: per-layer means with error bands
        ax = axes[0]
        means_a = [np.mean(values_a[l]) for l in layers_a if values_a[l]]
        stds_a = [np.std(values_a[l]) for l in layers_a if values_a[l]]
        valid_layers_a = [l for l in layers_a if values_a[l]]

        means_b = [np.mean(values_b[l]) for l in layers_b if values_b[l]]
        stds_b = [np.std(values_b[l]) for l in layers_b if values_b[l]]
        valid_layers_b = [l for l in layers_b if values_b[l]]

        # Normalize layer indices to [0, 1] for comparison across different depths
        norm_a = [l / max(layers_a) for l in valid_layers_a]
        norm_b = [l / max(layers_b) for l in valid_layers_b]

        ax.plot(norm_a, means_a, "b-", linewidth=2, label=f"{meta_a['tag']} ({meta_a['num_layers']}L)")
        ax.fill_between(norm_a,
                        [m - s for m, s in zip(means_a, stds_a)],
                        [m + s for m, s in zip(means_a, stds_a)],
                        alpha=0.2, color="blue")
        ax.plot(norm_b, means_b, "r-", linewidth=2, label=f"{meta_b['tag']} ({meta_b['num_layers']}L)")
        ax.fill_between(norm_b,
                        [m - s for m, s in zip(means_b, stds_b)],
                        [m + s for m, s in zip(means_b, stds_b)],
                        alpha=0.2, color="red")

        ax.set_xlabel("Relative Depth (0=first layer, 1=last)")
        ax.set_ylabel(stat_name)
        ax.set_title(f"{stat_name} by Layer Depth")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: distribution histograms
        ax = axes[1]
        ax.hist(all_a, bins=50, alpha=0.5, density=True, color="blue",
                label=f"{meta_a['tag']}")
        ax.hist(all_b, bins=50, alpha=0.5, density=True, color="red",
                label=f"{meta_b['tag']}")
        ax.set_xlabel(stat_name)
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of {stat_name} (all layers)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"compare_{stat_name}.png", dpi=150)
        plt.close()

    return results


# ---------------------------------------------------------------------------
# 2. Trajectory analysis
# ---------------------------------------------------------------------------

def trajectory_analysis(meta_a, fp_a, meta_b, fp_b, output_dir: Path):
    """
    Analyze the "trajectory" of activation statistics across layers.
    This is the layer-to-layer evolution that TaT-style analysis suggests
    carries information about reasoning validity.
    """
    print("\n" + "="*70)
    print("TRAJECTORY ANALYSIS")
    print("="*70)

    def build_trajectory_vectors(fingerprints, layers):
        """
        For each prompt, build a vector of [sparsity, gini, l1l2, energy_conc]
        at each layer, then measure layer-to-layer displacement.
        """
        features = ["sparsity_ratio", "gini_mean", "l1_l2_ratio",
                     "energy_concentration_top1pct"]
        trajectories = []
        displacements = []

        for fp in fingerprints:
            traj = []
            for l in layers:
                layer_data = fp["layers"].get(str(l), {})
                vec = [layer_data.get(f, 0.0) for f in features]
                traj.append(vec)

            traj = np.array(traj)  # [num_layers, num_features]
            trajectories.append(traj)

            # Compute layer-to-layer displacements
            if len(traj) > 1:
                disp = np.diff(traj, axis=0)  # [num_layers-1, num_features]
                displacements.append(disp)

        return np.array(trajectories), np.array(displacements) if displacements else None

    layers_a = sorted(int(k) for k in fp_a[0]["layers"].keys())
    layers_b = sorted(int(k) for k in fp_b[0]["layers"].keys())

    traj_a, disp_a = build_trajectory_vectors(fp_a, layers_a)
    traj_b, disp_b = build_trajectory_vectors(fp_b, layers_b)

    # --- Trajectory curvature ---
    # Higher curvature = more refinement per layer
    if disp_a is not None and disp_b is not None:
        curvature_a = np.linalg.norm(disp_a, axis=-1)  # [prompts, layers-1]
        curvature_b = np.linalg.norm(disp_b, axis=-1)

        mean_curv_a = curvature_a.mean(axis=0)  # per layer
        mean_curv_b = curvature_b.mean(axis=0)

        print(f"\n  Mean trajectory curvature:")
        print(f"    {meta_a['tag']}: {curvature_a.mean():.6f} ± {curvature_a.std():.6f}")
        print(f"    {meta_b['tag']}: {curvature_b.mean():.6f} ± {curvature_b.std():.6f}")

        # Plot curvature by layer
        fig, ax = plt.subplots(figsize=(10, 5))
        norm_layers_a = np.linspace(0, 1, len(mean_curv_a))
        norm_layers_b = np.linspace(0, 1, len(mean_curv_b))

        ax.plot(norm_layers_a, mean_curv_a, "b-", linewidth=2,
                label=f"{meta_a['tag']} ({meta_a['num_layers']}L)")
        ax.plot(norm_layers_b, mean_curv_b, "r-", linewidth=2,
                label=f"{meta_b['tag']} ({meta_b['num_layers']}L)")
        ax.set_xlabel("Relative Depth")
        ax.set_ylabel("Displacement Magnitude")
        ax.set_title("Layer-to-Layer Trajectory Displacement")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "trajectory_curvature.png", dpi=150)
        plt.close()

    # --- Trajectory effective dimensionality (PCA) ---
    print("\n  Trajectory effective dimensionality (PCA):")
    for label, traj, meta in [
        ("A", traj_a, meta_a), ("B", traj_b, meta_b)
    ]:
        # Stack all prompts' trajectories: [prompts, layers, features] -> [prompts*layers, features]
        flat = traj.reshape(-1, traj.shape[-1])
        flat_centered = flat - flat.mean(axis=0)

        try:
            _, singular_values, _ = np.linalg.svd(flat_centered, full_matrices=False)
            explained = singular_values**2 / (singular_values**2).sum()
            cum_explained = np.cumsum(explained)
            dims_95 = np.searchsorted(cum_explained, 0.95) + 1

            print(f"    {meta['tag']}: {dims_95} components for 95% variance "
                  f"(of {traj.shape[-1]} features)")
            print(f"      Singular value spectrum: {singular_values[:5].round(4)}")
        except Exception as e:
            print(f"    {meta['tag']}: SVD failed ({e})")


# ---------------------------------------------------------------------------
# 3. Classification — can we tell them apart?
# ---------------------------------------------------------------------------

def classification_analysis(meta_a, fp_a, meta_b, fp_b, output_dir: Path):
    """
    Train a simple classifier to distinguish model A from model B
    based on activation fingerprints. High accuracy = easy to tell apart.
    """
    print("\n" + "="*70)
    print("CLASSIFICATION ANALYSIS")
    print("="*70)

    features_to_use = [
        "sparsity_ratio", "gini_mean", "gini_std", "l1_l2_ratio",
        "mag_mean", "mag_std", "mag_skew", "mag_kurtosis",
        "energy_concentration_top1pct",
    ]

    def fingerprint_to_feature_vector(fp, num_layers=None):
        """Convert a single prompt's fingerprint into a flat feature vector."""
        layers = sorted(int(k) for k in fp["layers"].keys())
        if num_layers:
            # Sample layers evenly to normalize for different model depths
            indices = np.linspace(0, len(layers) - 1, num_layers, dtype=int)
            layers = [layers[i] for i in indices]

        vec = []
        for l in layers:
            layer_data = fp["layers"].get(str(l), {})
            for feat in features_to_use:
                vec.append(layer_data.get(feat, 0.0))
        return vec

    # Normalize to same number of "depth samples" for fair comparison
    num_depth_samples = 16  # Sample 16 evenly-spaced layers from each model

    X_a = [fingerprint_to_feature_vector(fp, num_depth_samples) for fp in fp_a]
    X_b = [fingerprint_to_feature_vector(fp, num_depth_samples) for fp in fp_b]

    X = np.array(X_a + X_b)
    y = np.array([0]*len(X_a) + [1]*len(X_b))

    # Check for NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n  Feature matrix shape: {X.shape}")
    print(f"  Class balance: {sum(y==0)} ({meta_a['tag']}) vs {sum(y==1)} ({meta_b['tag']})")

    # --- Logistic Regression ---
    print("\n  Logistic Regression (5-fold CV):")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(lr, X, y, cv=cv, scoring="accuracy")
    print(f"    Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    auc_scores = cross_val_score(lr, X, y, cv=cv, scoring="roc_auc")
    print(f"    AUC:      {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")

    # --- Random Forest ---
    print("\n  Random Forest (5-fold CV):")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores_rf = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
    print(f"    Accuracy: {scores_rf.mean():.4f} ± {scores_rf.std():.4f}")

    auc_scores_rf = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    print(f"    AUC:      {auc_scores_rf.mean():.4f} ± {auc_scores_rf.std():.4f}")

    # --- Feature importance ---
    rf.fit(X, y)
    importances = rf.feature_importances_

    # Map importances back to (layer_sample, feature_name) pairs
    feat_names = []
    for l_idx in range(num_depth_samples):
        for feat in features_to_use:
            feat_names.append(f"L{l_idx}/{feat}")

    sorted_idx = np.argsort(importances)[::-1]

    print("\n  Top 15 most discriminative features:")
    for rank, idx in enumerate(sorted_idx[:15]):
        print(f"    {rank+1:2d}. {feat_names[idx]:40s} importance={importances[idx]:.4f}")

    # --- Feature importance plot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    top_n = min(25, len(sorted_idx))
    top_idx = sorted_idx[:top_n]
    ax.barh(range(top_n), importances[top_idx], color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feat_names[i] for i in top_idx], fontsize=8)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Most Discriminative Fingerprint Features")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close()

    # --- Per-layer discriminability ---
    print("\n  Per-layer discriminability (single-layer logistic regression):")
    layer_accuracies = []
    for l_idx in range(num_depth_samples):
        start = l_idx * len(features_to_use)
        end = start + len(features_to_use)
        X_layer = X[:, start:end]

        lr_layer = LogisticRegression(max_iter=500, random_state=42)
        scores_layer = cross_val_score(lr_layer, X_layer, y, cv=cv, scoring="accuracy")
        layer_accuracies.append(scores_layer.mean())

    print(f"    Layer accuracies: {[f'{a:.3f}' for a in layer_accuracies]}")
    print(f"    Best single layer: sample {np.argmax(layer_accuracies)} "
          f"(accuracy={max(layer_accuracies):.4f})")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(num_depth_samples), layer_accuracies, color="steelblue")
    ax.axhline(y=0.5, color="red", linestyle="--", label="Chance level")
    ax.set_xlabel("Layer Sample Index (0=shallowest, 15=deepest)")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("Per-Layer Discriminability Between Models")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "per_layer_accuracy.png", dpi=150)
    plt.close()

    return {
        "logistic_regression_accuracy": float(scores.mean()),
        "logistic_regression_auc": float(auc_scores.mean()),
        "random_forest_accuracy": float(scores_rf.mean()),
        "random_forest_auc": float(auc_scores_rf.mean()),
        "per_layer_accuracies": [float(a) for a in layer_accuracies],
    }


# ---------------------------------------------------------------------------
# 4. Top-k neuron overlap analysis
# ---------------------------------------------------------------------------

def neuron_overlap_analysis(meta_a, fp_a, meta_b, fp_b, output_dir: Path):
    """
    Compare which neurons are most active in each model.
    Different models should activate different neuron populations.
    """
    print("\n" + "="*70)
    print("NEURON POPULATION OVERLAP ANALYSIS")
    print("="*70)

    # Only compare at layers that exist in both models
    layers_a = set(int(k) for k in fp_a[0]["layers"].keys())
    layers_b = set(int(k) for k in fp_b[0]["layers"].keys())

    # Use relative depth mapping
    def get_top_neurons(fingerprints, layer_idx):
        """Aggregate top-100 neuron indices across all prompts for a given layer."""
        from collections import Counter
        counter = Counter()
        for fp in fingerprints:
            indices = fp["layers"].get(str(layer_idx), {}).get("top_100_neuron_indices", [])
            counter.update(indices)
        return counter

    # Sample 5 evenly spaced layers from each model
    sample_a = [sorted(layers_a)[int(i)] for i in np.linspace(0, len(layers_a)-1, 5)]
    sample_b = [sorted(layers_b)[int(i)] for i in np.linspace(0, len(layers_b)-1, 5)]

    print("\n  Jaccard similarity of top-100 active neurons at matched depth positions:")

    for depth_label, la, lb in zip(
        ["early", "early-mid", "mid", "late-mid", "late"],
        sample_a, sample_b
    ):
        counter_a = get_top_neurons(fp_a, la)
        counter_b = get_top_neurons(fp_b, lb)

        # Top 100 most frequently active neurons
        top_a = set(n for n, _ in counter_a.most_common(100))
        top_b = set(n for n, _ in counter_b.most_common(100))

        if top_a and top_b:
            jaccard = len(top_a & top_b) / len(top_a | top_b)
            overlap = len(top_a & top_b)
            print(f"    {depth_label:10s}: Jaccard={jaccard:.4f}, "
                  f"overlap={overlap}/100 "
                  f"(A layer {la}, B layer {lb})")
        else:
            print(f"    {depth_label:10s}: insufficient data")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare SwiGLU activation fingerprints between two models"
    )
    parser.add_argument("--model-a", type=str, required=True,
                        help="Path to model A fingerprint directory")
    parser.add_argument("--model-b", type=str, required=True,
                        help="Path to model B fingerprint directory")
    parser.add_argument("--output", type=str, default="./analysis_results",
                        help="Output directory for plots and results")

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    meta_a, fp_a = load_fingerprints(args.model_a)
    meta_b, fp_b = load_fingerprints(args.model_b)

    # Ensure we compare the same number of prompts
    n = min(len(fp_a), len(fp_b))
    fp_a = fp_a[:n]
    fp_b = fp_b[:n]
    print(f"\nComparing {n} prompts from each model")

    # Run analyses
    stat_results = compare_per_layer_stats(meta_a, fp_a, meta_b, fp_b, output_dir)
    trajectory_analysis(meta_a, fp_a, meta_b, fp_b, output_dir)
    class_results = classification_analysis(meta_a, fp_a, meta_b, fp_b, output_dir)
    neuron_overlap_analysis(meta_a, fp_a, meta_b, fp_b, output_dir)

    # --- Summary ---
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n  Models compared: {meta_a['tag']} vs {meta_b['tag']}")
    print(f"  Prompts compared: {n}")
    print(f"\n  Classification accuracy (Random Forest): "
          f"{class_results['random_forest_accuracy']:.4f}")
    print(f"  Classification AUC (Random Forest): "
          f"{class_results['random_forest_auc']:.4f}")

    print(f"\n  Key separations (Cohen's d):")
    for stat_name in ["sparsity_ratio", "gini_mean", "l1_l2_ratio",
                      "energy_concentration_top1pct"]:
        d = stat_results.get(stat_name, {}).get("cohens_d", "N/A")
        print(f"    {stat_name:35s}: {d}")

    verdict = class_results["random_forest_accuracy"]
    print(f"\n  VERDICT: ", end="")
    if verdict > 0.95:
        print("STRONGLY DISTINGUISHABLE — fingerprinting looks very promising")
    elif verdict > 0.85:
        print("CLEARLY DISTINGUISHABLE — fingerprinting has potential, worth pursuing")
    elif verdict > 0.70:
        print("MODERATELY DISTINGUISHABLE — some signal, needs refinement")
    elif verdict > 0.55:
        print("WEAKLY DISTINGUISHABLE — marginal signal, approach may need rethinking")
    else:
        print("NOT DISTINGUISHABLE — fingerprinting does not work for this model pair")

    # Save summary
    summary = {
        "model_a": meta_a,
        "model_b": meta_b,
        "num_prompts": n,
        "classification": class_results,
        "statistical_separations": {
            k: {
                "cohens_d": v.get("cohens_d"),
                "ks_statistic": v.get("ks_statistic"),
                "ks_pvalue": v.get("ks_pvalue"),
            }
            for k, v in stat_results.items()
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  All results saved to {output_dir}/")


if __name__ == "__main__":
    main()
