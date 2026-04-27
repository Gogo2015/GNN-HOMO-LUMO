# PCQM4Mv2 — GNN for HOMO-LUMO Gap Prediction

CS 7643 final project — Net-Runners.

---

## Setup (PACE, one-time)

Connect to GT VPN, then SSH in:

```bash
ssh <gtid>@login-ice.pace.gatech.edu
```

**1. Clone into scratch**

```bash
cd ~/scratch
git clone <repo-url> DL-NetRunners
cd DL-NetRunners
```

**2. Create the conda environment**

```bash
module load anaconda3/2022.05.0.1
conda env create --prefix ~/scratch/dl-netrunners -f environment/environment.yml
conda activate ~/scratch/dl-netrunners
```

**3. Download the dataset (~10 GB)**

Run this on an interactive node, not the login node:

```bash
salloc -N1 -t 0:30:00 --cpus-per-task 4 --ntasks-per-node=1 --mem=16G
module load anaconda3/2022.05.0.1 && conda activate ~/scratch/dl-netrunners
cd ~/scratch/DL-NetRunners
python3 -m src.download_data --data_root ~/scratch/data
```

---

## Running Experiments

Submit both jobs from the login node:

```bash
sbatch jobs/train_baseline.sh
sbatch jobs/train_virtual_node.sh
```

Each job runs up to 8 hours and checkpoints every epoch. To resume after a wall-time cutoff, add `--resume` to the `python3` line in the job script and resubmit.

**Monitor / check logs:**

```bash
squeue -u <gtid>
tail -f logs/baseline_<jobid>.out
```

**Checkpoints** are saved to `~/scratch/checkpoints/`:
- `last_gcn.pt` / `last_vn.pt` — latest epoch (used for resuming)
- `best_gcn.pt` / `best_vn.pt` — best validation MAE

---

## Project Layout

```
configs/
  gcn_baseline.yml       # baseline config
  virtual_node.yml       # virtual node config
environment/
  environment.yml        # conda env spec
jobs/
  train_baseline.sh      # SLURM script — baseline
  train_virtual_node.sh  # SLURM script — virtual node
src/
  download_data.py       # dataset download
  train.py               # training + validation loop
  utils.py               # seed, config, checkpoint helpers
  models/
    gnn.py               # AtomEncoder + GINELayers + MLP head
    layers.py            # GINELayer (batch norm + residual)
    virtual_node.py      # virtual node module
```

---

## Contributions

| Component | Gaurav | Kapil | Kartikey | Archith |
|---|---|---|---|---|
| Repo + environment setup | Lead | | | |
| Data pipeline (`download_data.py`) | Lead | | | |
| GIN layers (`layers.py`) | Lead | | | |
| Virtual node (`virtual_node.py`) | Lead | | | |
| Top-level model (`gnn.py`) | Lead | | | |
| Training loop (`train.py`) | Lead | | | |
| Experiment runs | | | | |
| Hyperparameter search | | | | |
| Dataset EDA + figures | | | | |
| Results plots & tables | | | | |
| Introduction | | | | |
| Related work | | | | |
| Methodology | | | | |
| Experiments writeup | | | | |
| Discussion / conclusion | | | | |
| Presentation slides | | | | |
| README | Lead | | | |
