import pandas as pd
import matplotlib.pyplot as plt
import os

ckpt_dir = "~/scratch/checkpoints/"

files = {
    "GCN (10k)": "history_gcn_sub10000.csv",
    "VN (10k)": "history_vn_sub10000.csv",
    "VN (50k)": "history_vn_sub50000.csv"
}

plt.figure(figsize=(10, 6))

for label, filename in files.items():
    path = os.path.join(ckpt_dir, filename)
    if os.path.exists(path):
        df = pd.read_csv(path)
        linewidth = 2.5 if "50k" in label else 1.5
        plt.plot(df['epoch'], df['val_mae'], label=label, linewidth=linewidth)
    else:
        print(f"Skipping {label}: {path} not found.")

plt.axhline(y=0.12, color='r', linestyle='--', label='Target (< 0.12 eV)')

plt.title('Model Convergence: Validation MAE Comparison', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MAE (eV)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('figure_convergence.png', dpi=300)
print("Successfully generated figure_convergence.png in the root directory!")
