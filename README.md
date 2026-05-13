# PCQM4Mv2 — GNN for Molecular HOMO-LUMO Gap Prediction

Graph neural network for predicting the HOMO-LUMO energy gap of molecules directly from 2D graph structure. CS 7643 final project, Team Net-Runners.

The goal is to replace expensive DFT calculations (hours per molecule) with a learned model that predicts the same scalar in milliseconds, for use in high-throughput materials and drug screening.

## Results

Both models trained for 5 epochs on the full PCQM4Mv2 training split (3.38M molecules) on an H100 GPU.

| Model              | Validation MAE (eV) | Parameters | Best Epoch |
|--------------------|---------------------|------------|------------|
| Baseline GINE      | **0.1421**          | 1.93M      | 5          |
| Virtual Node GINE  | 0.1487              | 2.53M      | 3          |

The Virtual Node did not improve aggregate MAE over the deeper GINE baseline. A size-bucket breakdown told a more nuanced story: the VN reduced MAE on molecules with 21–40 atoms (where long-range information flow matters more), but performed slightly worse on smaller molecules and was noisy in the 41+ bucket. See the [report](report/Final_Report.pdf) for full analysis.

### Size-bucket MAE

| Size Bucket (# atoms) | Baseline GINE | VN GINE     |
|-----------------------|---------------|-------------|
| 0–10                  | **0.1590**    | 0.1601      |
| 11–20                 | **0.1337**    | 0.1422      |
| 21–30                 | 0.2314        | **0.2133**  |
| 31–40                 | 0.5078        | **0.4761**  |

## Architecture

Both models share the same backbone:

- Atom and bond categorical features → learned embeddings (hidden dim 256)
- 7 stacked GINE layers (each: GINE conv → BatchNorm → ReLU → Dropout → residual)
- Per-layer atom embeddings mean-pooled to graph level
- Sum across layers (jumping knowledge) → 2-layer MLP regression head

The GINE update rule:

$$h_v^{(l+1)} = \text{MLP}^{(l)}\!\left((1+\epsilon^{(l)})\, h_v^{(l)} + \sum_{u \in N(v)} \text{ReLU}(h_u^{(l)} + e_{vu}^{(l)})\right)$$

**Virtual Node variant** adds a single global node $g$ shared across the molecule. Before each layer the VN embedding is broadcast and added to every atom embedding; after the layer it is updated by summing atom embeddings and passing through a 2-layer MLP. This gives every atom a constant-hop pathway to every other atom.

## Setup

```bash
conda env create -f environment/environment.yml
conda activate gnn-mol
```

First run downloads PCQM4Mv2 (~13 GB compressed) into `data/`.

## Training

```bash
# Baseline GINE
python train.py --config configs/gine_baseline.yaml

# Virtual Node variant
python train.py --config configs/gine_vn.yaml
```

Both configs use batch size 256, learning rate 3e-4, Adam, dropout 0.1, L1 loss (matches the official PCQM4Mv2 metric). Each run takes ~3 hours per epoch on an H100.

## Evaluation

```bash
python test.py --checkpoint runs/<run_name>/best.pt
```

Outputs: validation MAE, parity plot, error distribution histogram, size-bucket breakdown.

## Hyperparameter Search

Search was run on a 200k/20k subset before committing to full-dataset training:

| Run            | Hidden | Layers | LR     | VN  | Subset MAE (eV) |
|----------------|--------|--------|--------|-----|-----------------|
| Base GINE      | 256    | 5      | 3e-4   | No  | 0.2302          |
| Narrow GINE    | 128    | 5      | 3e-4   | No  | 0.2402          |
| **Deeper GINE**| 256    | 7      | 3e-4   | No  | **0.2171**      |
| Low-LR GINE    | 256    | 5      | 1e-4   | No  | 0.2413          |
| VN Pilot       | 256    | 5      | 3e-4   | Yes | 0.2347          |

Depth helped most; narrowing or lowering LR hurt. The 7-layer architecture was selected for the final comparison.

## Repository Structure

```
.
├── configs/              # YAML configs for each run
├── src/
│   ├── data.py           # PCQM4Mv2 loader, smiles2graph
│   ├── layers.py         # GINE conv layer
│   ├── gnn.py            # Model definitions (baseline + VN)
│   └── train.py          # Training loop
├── test.py               # Evaluation + plots
├── environment/
└── report/               # Final report PDF + figures
```

## Limitations

- Only 2D graph connectivity is used. HOMO-LUMO gaps depend on 3D electronic structure; the dataset provides 3D conformer coordinates we didn't incorporate.
- 5-epoch training was a compute compromise. Neither model showed overfitting at epoch 5, so longer training should improve both.
- Single seed per final run — the 0.0066 eV gap between models should be read as suggestive rather than statistically conclusive.

## References

Key papers:
- Hu et al., *OGB-LSC: A Large-Scale Challenge for ML on Graphs*, NeurIPS 2021
- Xu et al., *How Powerful are Graph Neural Networks?*, ICLR 2019
- Alon & Yahav, *On the Bottleneck of Graph Neural Networks*, ICLR 2021
- Gilmer et al., *Neural Message Passing for Quantum Chemistry*, ICML 2017

## Team

- Kapil Meena — infrastructure, experiment execution support
- Gaurav Mitra — custom GNN implementation, training loop
- Archith K. Iyer — hyperparameter tuning, full-dataset training, evaluation, visualizations