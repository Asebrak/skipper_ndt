"""
ÉTAPE 1 - EXPLORATION DES DONNÉES
Skipper NDT x HETIC
Exécuter depuis : /Users/assemelabrak/Downloads/Projects/pipelines_skipper/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR  = Path("Training_database_float16")
CSV_PATH  = Path("pipe_detection_label.csv")          # ajuster si besoin
OUT_DIR   = Path("exploration_outputs")
OUT_DIR.mkdir(exist_ok=True)

CHANNELS  = ["Bx", "By", "Bz", "Norm"]

# ── 1. CHARGER LE CSV ────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, sep=";")
print("=" * 60)
print(f"Dataset : {len(df)} échantillons, {df.shape[1]} colonnes")
print("=" * 60)
print(df.dtypes)
print()

# ── 2. STATISTIQUES DU CSV ───────────────────────────────────────────────────
print("── Label (Tâche 1 : présence de conduite) ──")
print(df["label"].value_counts().rename({0: "Absent", 1: "Présent"}))
print()

print("── pipe_type (Tâche 4 : conduites parallèles) ──")
print(df["pipe_type"].value_counts())
print()

print("── width_m (Tâche 2 : largeur magnétique) ──")
print(df["width_m"].describe().round(2))
print()

print("── coverage_type (Tâche 3 : intensité courant) ──")
print(df["coverage_type"].value_counts())
print()

print("── noisy / noise_type ──")
print(df["noisy"].value_counts())
print(df["noise_type"].value_counts())
print()

# ── 3. INSPECTER 1 FICHIER NPZ ───────────────────────────────────────────────
sample_file = DATA_DIR / df["field_file"].iloc[0]
print(f"── Inspection de : {sample_file.name} ──")
with np.load(sample_file) as npz:
    keys = list(npz.keys())
    print(f"  Clés : {keys}")
    for k in keys:
        arr = npz[k]
        print(f"  {k:20s} shape={arr.shape}  dtype={arr.dtype}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}")
print()

# ── 4. VÉRIFIER LES SHAPES SUR UN SOUS-ENSEMBLE ──────────────────────────────
print("── Dimensions des 200 premiers échantillons ──")
shapes = []
for fname in df["field_file"].iloc[:200]:
    path = DATA_DIR / fname
    if path.exists():
        with np.load(path) as npz:
            key = list(npz.keys())[0]
            shapes.append(npz[key].shape)
    else:
        shapes.append(None)

valid_shapes = [s for s in shapes if s is not None]
heights = [s[-2] for s in valid_shapes]
widths  = [s[-1] for s in valid_shapes]
print(f"  Hauteurs : min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")
print(f"  Largeurs : min={min(widths)},  max={max(widths)},  mean={np.mean(widths):.0f}")
print(f"  Fichiers manquants : {shapes.count(None)}")

# ── 5. VISUALISER 6 EXEMPLES ─────────────────────────────────────────────────
def load_npz(fname):
    path = DATA_DIR / fname
    with np.load(path) as npz:
        key = list(npz.keys())[0]
        data = npz[key].astype(np.float32)
    # Assurer shape (4, H, W)
    if data.ndim == 2:
        data = data[np.newaxis]  # (1, H, W)
    if data.ndim == 3 and data.shape[0] not in [1, 4]:
        data = data.transpose(2, 0, 1)  # (H,W,C) → (C,H,W)
    return data

# Sélectionner 3 avec conduite, 3 sans
samples_1 = df[df["label"] == 1].sample(3, random_state=42)
samples_0 = df[df["label"] == 0].sample(3, random_state=42)
selected  = pd.concat([samples_1, samples_0])

fig, axes = plt.subplots(6, 4, figsize=(16, 20))
fig.suptitle("Exploration : 3 avec conduite / 3 sans (canaux Bx, By, Bz, Norm)",
             fontsize=14, fontweight="bold")

for row, (_, meta) in enumerate(selected.iterrows()):
    data = load_npz(meta["field_file"])
    n_ch = data.shape[0]
    label_str = "✅ Conduite" if meta["label"] == 1 else "❌ Pas de conduite"
    for col in range(4):
        ax = axes[row, col]
        if col < n_ch:
            img = data[col]
            im  = ax.imshow(img, cmap="RdBu_r", aspect="auto",
                            vmin=np.percentile(img, 2), vmax=np.percentile(img, 98))
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(f"{CHANNELS[col]}\n{label_str}" if col == 0
                         else CHANNELS[col], fontsize=8)
        else:
            ax.axis("off")
        ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig(OUT_DIR / "01_visualisation_echantillons.png", dpi=120)
print(f"\nFigure sauvegardée → {OUT_DIR}/01_visualisation_echantillons.png")

# ── 6. DISTRIBUTIONS DES LABELS ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Distribution des labels", fontsize=13, fontweight="bold")

# T1 : présence
axes[0].bar(["Absent (0)", "Présent (1)"],
            df["label"].value_counts().sort_index().values,
            color=["#e74c3c", "#2ecc71"], edgecolor="k")
axes[0].set_title("Tâche 1 : Présence de conduite")
axes[0].set_ylabel("Nombre d'échantillons")
for bar in axes[0].patches:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 str(int(bar.get_height())), ha="center", fontsize=11)

# T2 : width_m
axes[1].hist(df["width_m"], bins=40, color="#3498db", edgecolor="k", alpha=0.8)
axes[1].set_title("Tâche 2 : Distribution width_m")
axes[1].set_xlabel("Largeur (m)")
axes[1].axvline(df["width_m"].mean(), color="red", linestyle="--",
                label=f"Moyenne = {df['width_m'].mean():.1f}m")
axes[1].legend()

# T4 : pipe_type
counts = df["pipe_type"].value_counts()
axes[2].bar(counts.index, counts.values,
            color=["#9b59b6", "#e67e22"], edgecolor="k")
axes[2].set_title("Tâche 4 : Single vs Parallel")
for bar in axes[2].patches:
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(int(bar.get_height())), ha="center", fontsize=11)

plt.tight_layout()
plt.savefig(OUT_DIR / "02_distributions_labels.png", dpi=120)
print(f"Figure sauvegardée → {OUT_DIR}/02_distributions_labels.png")

plt.show()
print("\n✅ Exploration terminée !")
