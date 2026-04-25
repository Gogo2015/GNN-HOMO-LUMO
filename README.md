# PCQM4Mv2 — GNN for HOMO-LUMO gap prediction

CS 7643 final project, Net-runners.

## Setup

```bash
conda env create -f environment/environment.yml
conda activate dl-netrunners
```

## Running an experiment

```bash
python -m src.train --config configs/gcn_baseline.yaml
```

## Project layout

```
environment/environment.yml    # conda env
src/
  download_data.py             # PCQM4Mv2 loaders
  train.py                     # training loop
  utils.py                     #
  models/
    gnn.py                     # full model
    layers.py                  # layers
    virtual_node.py            # virtual node
configs/                       # configs for experiments
```

## Contributions

| Component | Gaurav | Kapil | Kartikey | Archith |
|---|---|---|---|---|
| Repository + environment setup | Lead | | | |
| Data pipeline (`src/data.py`) | | | | |
| Custom GCN / GIN layers (`layers.py`) | | | | |
| Virtual node module (`virtual_node.py`) | | | | |
| Top-level model (`gnn.py`) | | | | |
| Training loop (`train.py`) | | | | |
| Experiment runs | | | | |
| Hyperparameter search | | | | |
| Dataset EDA + statistics figures | | | | |
| Results plots & tables | |  | | |
| Introduction | | |  | |
| Related work | | | | |
| Methodology |  | |  | |
| Experiments writeup |  |  | |  |
| Discussion / conclusion |  | |  | |
| Presentation slides | | |  | |
| README | Lead | | | |
