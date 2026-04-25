import argparse
import os
from ogb.lsc import PygPCQM4Mv2Dataset
from ogb.utils import smiles2graph


def get_dataset(root: str):
    return PygPCQM4Mv2Dataset(root=root, smiles2graph=smiles2graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.expanduser("~/scratch/data"))
    args = parser.parse_args()

    dataset = get_dataset(args.data_root)
    split = dataset.get_idx_split()

    print(f"Dataset size : {len(dataset):,}")
    print(f"Train        : {len(split['train']):,}")
    print(f"Val          : {len(split['valid']):,}")
    print(f"Test         : {len(split['test-dev']):,}")
    
    sample = dataset[0]
    print(f"Node features : {sample.x.shape}")
    print(f"Edge features : {sample.edge_attr.shape}")
    print(f"Label         : {sample.y.item():.4f} eV")
