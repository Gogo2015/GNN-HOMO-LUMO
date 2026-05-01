import argparse
from pathlib import Path
import csv

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator
from ogb.utils import smiles2graph

from .models.gnn import GNN
from .utils import set_seed, load_config, save_checkpoint, load_checkpoint


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(batch).view(-1)
        y = batch.y.view(-1)

        loss = F.l1_loss(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, evaluator, device):
    model.eval()
    y_true, y_pred = [], []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)

        y_true.append(batch.y.view(-1).cpu())
        y_pred.append(pred.view(-1).cpu())

    return evaluator.eval({
        "y_true": torch.cat(y_true),
        "y_pred": torch.cat(y_pred)
    })["mae"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train_subset", type=int, default=None,
                        help="Use only first N train molecules")
    parser.add_argument("--val_subset", type=int, default=None,
                        help="Use only first N validation molecules")
    args = parser.parse_args()

    cfg = load_config(args.config)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt_dir = Path(cfg.get("checkpoint_dir", "/workspace/checkpoints")).expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load dataset
    # ----------------------------
    dataset = PygPCQM4Mv2Dataset(
        root=Path(cfg.get("data_root", "/workspace/data")).expanduser(),
        smiles2graph=smiles2graph,
    )
    split = dataset.get_idx_split()

    train_idx = split["train"]
    val_idx = split["valid"]

    if args.train_subset is not None:
        train_idx = train_idx[:args.train_subset]

    if args.val_subset is not None:
        val_idx = val_idx[:args.val_subset]

    print(f"Train molecules: {len(train_idx):,}")
    print(f"Val molecules:   {len(val_idx):,}")

    train_loader = DataLoader(
        dataset[train_idx],
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        dataset[val_idx],
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=4
    )

    # ----------------------------
    # Build model
    # ----------------------------
    model = GNN(
        num_layers=cfg["num_layers"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        use_virtual_node=cfg["use_virtual_node"],
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["scheduler_step"],
        gamma=cfg["scheduler_gamma"]
    )

    evaluator = PCQM4Mv2Evaluator()

    tag = "vn" if cfg["use_virtual_node"] else "gcn"

    # if subset run, save separately so you don't overwrite full checkpoints
    if args.train_subset is not None:
        tag += f"_sub{args.train_subset}"

    last_ckpt = ckpt_dir / f"last_{tag}.pt"
    best_ckpt = ckpt_dir / f"best_{tag}.pt"
    history_csv = ckpt_dir / f"history_{tag}.csv"

    start_epoch = 1
    best_val_mae = float("inf")

    if args.resume and last_ckpt.exists():
        loaded_epoch, best_val_mae = load_checkpoint(last_ckpt, model, optimizer, scheduler)
        start_epoch = loaded_epoch + 1
        print(f"Resumed from epoch {loaded_epoch}, best val MAE so far: {best_val_mae:.4f}")

    # initialize csv log
    if start_epoch == 1:
        with open(history_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_mae"])

    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_mae = eval_epoch(model, val_loader, evaluator, device)
        scheduler.step()

        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | val_mae {val_mae:.4f}")

        # log csv
        with open(history_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_mae])

        save_checkpoint(model, optimizer, scheduler, epoch, val_mae, last_ckpt)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_checkpoint(model, optimizer, scheduler, epoch, val_mae, best_ckpt)
            print(f"  -> New best val MAE: {best_val_mae:.4f}")

    print(f"\nBest val MAE ({tag}): {best_val_mae:.4f} eV")
    print(f"Saved training history to {history_csv}")


if __name__ == "__main__":
    main()