import argparse
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator
from ogb.utils import smiles2graph

from .models.gnn import GNN
from .utils import load_config, load_checkpoint


@torch.no_grad()
def eval_model(model, loader, evaluator, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y_true.append(batch.y.view(-1).cpu())
        y_pred.append(pred.view(-1).cpu())
    return evaluator.eval({"y_true": torch.cat(y_true), "y_pred": torch.cat(y_pred)})["mae"]


def load_model(cfg, ckpt_path, device):
    model = GNN(
        num_layers=cfg["num_layers"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        use_virtual_node=cfg["use_virtual_node"],
    ).to(device)
    epoch, val_mae = load_checkpoint(ckpt_path, model)
    return model, epoch, val_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcn_config", required=True, help="Config for baseline GCN")
    parser.add_argument("--vn_config", required=True, help="Config for virtual node model")
    parser.add_argument("--gcn_checkpoint", default=None)
    parser.add_argument("--vn_checkpoint", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    gcn_cfg = load_config(args.gcn_config)
    vn_cfg = load_config(args.vn_config)

    gcn_ckpt_dir = Path(gcn_cfg.get("checkpoint_dir", "~/scratch/checkpoints")).expanduser()
    vn_ckpt_dir = Path(vn_cfg.get("checkpoint_dir", "~/scratch/checkpoints")).expanduser()

    gcn_ckpt = Path(args.gcn_checkpoint) if args.gcn_checkpoint else gcn_ckpt_dir / "best_gcn.pt"
    vn_ckpt = Path(args.vn_checkpoint) if args.vn_checkpoint else vn_ckpt_dir / "best_vn.pt"

    # Both configs use the same dataset, use gcn_cfg for data root
    dataset = PygPCQM4Mv2Dataset(
        root=Path(gcn_cfg.get("data_root", "~/scratch/data")).expanduser(),
        smiles2graph=smiles2graph,
    )
    split = dataset.get_idx_split()
    val_loader = DataLoader(
        dataset[split["valid"]],
        batch_size=gcn_cfg["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    evaluator = PCQM4Mv2Evaluator()

    gcn_model, gcn_epoch, _ = load_model(gcn_cfg, gcn_ckpt, device)
    vn_model, vn_epoch, _ = load_model(vn_cfg, vn_ckpt, device)

    print(f"GCN baseline    — checkpoint epoch {gcn_epoch}, params: {sum(p.numel() for p in gcn_model.parameters()):,}")
    print(f"Virtual node    — checkpoint epoch {vn_epoch},  params: {sum(p.numel() for p in vn_model.parameters()):,}")
    print()

    gcn_mae = eval_model(gcn_model, val_loader, evaluator, device)
    vn_mae = eval_model(vn_model, val_loader, evaluator, device)

    print("=" * 40)
    print(f"{'Model':<20} {'Val MAE (eV)':>12}")
    print("-" * 40)
    print(f"{'GCN baseline':<20} {gcn_mae:>12.4f}")
    print(f"{'Virtual node':<20} {vn_mae:>12.4f}")
    print("=" * 40)
    delta = gcn_mae - vn_mae
    winner = "virtual node" if delta > 0 else "GCN baseline"
    print(f"Delta: {abs(delta):.4f} eV  →  {winner} wins")
    print(f"Target MAE < 0.12 eV: GCN {'PASS' if gcn_mae < 0.12 else 'FAIL'}  |  VN {'PASS' if vn_mae < 0.12 else 'FAIL'}")


if __name__ == "__main__":
    main()
