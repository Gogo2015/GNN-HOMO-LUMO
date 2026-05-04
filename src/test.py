import argparse
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from ogb.lsc import PygPCQM4Mv2Dataset
from ogb.utils import smiles2graph

from .models.gnn import GNN
from .utils import load_config


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    y_true, y_pred, mol_sizes = [], [], []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)

        y_true.extend(batch.y.view(-1).cpu().tolist())
        y_pred.extend(pred.view(-1).cpu().tolist())

        counts = torch.bincount(batch.batch.cpu())
        mol_sizes.extend(counts.tolist())

    return y_true, y_pred, mol_sizes


def load_model(cfg, ckpt_path, device):
    model = GNN(
        num_layers=cfg["num_layers"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        use_virtual_node=cfg["use_virtual_node"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    return model, ckpt["epoch"], ckpt["val_mae"]


def compute_mae(y_true, y_pred):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    return torch.mean(torch.abs(y_true - y_pred)).item()


def bucketize_sizes(sizes, errors):
    buckets = {
        "0-10": [],
        "11-20": [],
        "21-30": [],
        "31-40": [],
        "41+": [],
    }

    for size, err in zip(sizes, errors):
        if size <= 10:
            buckets["0-10"].append(err)
        elif size <= 20:
            buckets["11-20"].append(err)
        elif size <= 30:
            buckets["21-30"].append(err)
        elif size <= 40:
            buckets["31-40"].append(err)
        else:
            buckets["41+"].append(err)

    return {
        key: (sum(vals) / len(vals) if len(vals) > 0 else 0.0)
        for key, vals in buckets.items()
    }


def save_mae_bar(gcn_mae, vn_mae, output_dir):
    plt.figure(figsize=(6, 5))
    plt.bar(["Baseline GINE", "Virtual Node GINE"], [gcn_mae, vn_mae])
    plt.ylabel("Validation MAE (eV)")
    plt.title("Final Validation MAE Comparison")
    plt.tight_layout()
    plt.savefig(output_dir / "final_mae_comparison.png", dpi=200)
    plt.close()


def save_parity_plot(y_true, y_pred, title, path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=5, alpha=0.25)

    lo = min(min(y_true), min(y_pred))
    hi = max(max(y_true), max(y_pred))
    plt.plot([lo, hi], [lo, hi], linestyle="--")

    plt.xlabel("True HOMO-LUMO Gap (eV)")
    plt.ylabel("Predicted HOMO-LUMO Gap (eV)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcn_config", required=True)
    parser.add_argument("--vn_config", required=True)
    parser.add_argument("--data_root", default="/workspace/data")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--output_dir", default="final_results")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(args.checkpoint_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gcn_cfg = load_config(args.gcn_config)
    vn_cfg = load_config(args.vn_config)

    dataset = PygPCQM4Mv2Dataset(
        root=args.data_root,
        smiles2graph=smiles2graph,
    )
    split = dataset.get_idx_split()

    val_loader = DataLoader(
        dataset[split["valid"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    gcn_model, gcn_epoch, gcn_ckpt_mae = load_model(
        gcn_cfg,
        checkpoint_dir / "best_gcn.pt",
        device,
    )

    vn_model, vn_epoch, vn_ckpt_mae = load_model(
        vn_cfg,
        checkpoint_dir / "best_vn.pt",
        device,
    )

    print(f"Loaded baseline checkpoint from epoch {gcn_epoch} with saved val MAE {gcn_ckpt_mae:.4f}")
    print(f"Loaded VN checkpoint from epoch {vn_epoch} with saved val MAE {vn_ckpt_mae:.4f}")

    gcn_true, gcn_pred, gcn_sizes = collect_predictions(gcn_model, val_loader, device)
    vn_true, vn_pred, vn_sizes = collect_predictions(vn_model, val_loader, device)

    gcn_mae = compute_mae(gcn_true, gcn_pred)
    vn_mae = compute_mae(vn_true, vn_pred)

    print(f"\nBaseline GINE Final Validation MAE: {gcn_mae:.4f} eV")
    print(f"Virtual Node GINE Final Validation MAE: {vn_mae:.4f} eV")

    if gcn_mae < vn_mae:
        print(f"Baseline wins by {vn_mae - gcn_mae:.4f} eV")
    else:
        print(f"Virtual Node wins by {gcn_mae - vn_mae:.4f} eV")

    pred_df = pd.DataFrame({
        "true_gap": gcn_true,
        "gcn_prediction": gcn_pred,
        "vn_prediction": vn_pred,
        "molecule_size_atoms": gcn_sizes,
        "gcn_abs_error": [abs(a - b) for a, b in zip(gcn_true, gcn_pred)],
        "vn_abs_error": [abs(a - b) for a, b in zip(vn_true, vn_pred)],
    })
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    save_mae_bar(gcn_mae, vn_mae, output_dir)

    save_parity_plot(
        gcn_true,
        gcn_pred,
        "Parity Plot - Baseline GINE",
        output_dir / "parity_gcn.png",
    )

    save_parity_plot(
        vn_true,
        vn_pred,
        "Parity Plot - Virtual Node GINE",
        output_dir / "parity_vn.png",
    )

    gcn_errors = pred_df["gcn_abs_error"].tolist()
    vn_errors = pred_df["vn_abs_error"].tolist()

    plt.figure(figsize=(7, 5))
    plt.hist(gcn_errors, bins=50, alpha=0.6, label="Baseline GINE")
    plt.hist(vn_errors, bins=50, alpha=0.6, label="Virtual Node GINE")
    plt.xlabel("Absolute Error (eV)")
    plt.ylabel("Number of Molecules")
    plt.title("Prediction Error Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "error_histogram.png", dpi=200)
    plt.close()

    gcn_bucket = bucketize_sizes(gcn_sizes, gcn_errors)
    vn_bucket = bucketize_sizes(vn_sizes, vn_errors)

    bucket_df = pd.DataFrame({
        "size_bucket": list(gcn_bucket.keys()),
        "gcn_mae": list(gcn_bucket.values()),
        "vn_mae": [vn_bucket[k] for k in gcn_bucket.keys()],
    })
    bucket_df.to_csv(output_dir / "size_bucket_mae.csv", index=False)

    labels = list(gcn_bucket.keys())

    plt.figure(figsize=(7, 5))
    plt.plot(labels, [gcn_bucket[k] for k in labels], marker="o", label="Baseline GINE")
    plt.plot(labels, [vn_bucket[k] for k in labels], marker="o", label="Virtual Node GINE")
    plt.xlabel("Molecule Size Bucket (# Atoms)")
    plt.ylabel("Average MAE (eV)")
    plt.title("MAE by Molecule Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "size_bucket_mae.png", dpi=200)
    plt.close()

    summary_df = pd.DataFrame({
        "model": ["Baseline GINE", "Virtual Node GINE"],
        "checkpoint_epoch": [gcn_epoch, vn_epoch],
        "checkpoint_saved_val_mae": [gcn_ckpt_mae, vn_ckpt_mae],
        "recomputed_val_mae": [gcn_mae, vn_mae],
    })
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)

    print(f"\nSaved outputs to: {output_dir.resolve()}")
    print("Generated:")
    print("  predictions.csv")
    print("  summary_metrics.csv")
    print("  size_bucket_mae.csv")
    print("  final_mae_comparison.png")
    print("  parity_gcn.png")
    print("  parity_vn.png")
    print("  error_histogram.png")
    print("  size_bucket_mae.png")


if __name__ == "__main__":
    main()