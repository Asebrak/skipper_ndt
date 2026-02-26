"""
VISUALISATION DES RÉSULTATS
Skipper NDT x HETIC

Usage : python plot_results.py
Génère des graphiques à partir des fichiers history.json de chaque tâche.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

TASKS = ["task1", "task2", "task3", "task4"]
TASK_NAMES = {
    "task1": "T1 : Présence de conduite",
    "task2": "T2 : Largeur magnétique",
    "task3": "T3 : Intensité du courant",
    "task4": "T4 : Conduites parallèles",
}
MODELS_DIR = Path("models")
OUT_DIR    = Path("exploration_outputs")
OUT_DIR.mkdir(exist_ok=True)


def load_history(task):
    path = MODELS_DIR / task / "history.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_results(task):
    path = MODELS_DIR / task / "results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── COURBES DE LOSS ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Courbes d'entraînement — Skipper NDT x HETIC", fontsize=14, fontweight="bold")

for ax, task in zip(axes.flat, TASKS):
    hist = load_history(task)
    if hist is None:
        ax.text(0.5, 0.5, f"Pas encore entraîné\n({task})",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title(TASK_NAMES[task])
        continue

    epochs = range(1, len(hist["train_loss"]) + 1)
    ax.plot(epochs, hist["train_loss"], label="Train loss", color="#3498db", linewidth=2)
    ax.plot(epochs, hist["val_loss"],   label="Val loss",   color="#e74c3c", linewidth=2)
    ax.set_title(TASK_NAMES[task])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # Annoter le minimum val loss
    min_idx = np.argmin(hist["val_loss"])
    ax.axvline(min_idx + 1, color="green", linestyle="--", alpha=0.6,
               label=f"Best epoch={min_idx+1}")

plt.tight_layout()
plt.savefig(OUT_DIR / "training_curves.png", dpi=120)
print(f"Sauvegardé → {OUT_DIR}/training_curves.png")


# ── TABLEAU DE RÉSULTATS ──────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.axis("off")

table_data = [["Tâche", "Métrique", "Valeur", "Objectif", "✅/❌"]]
objectives = {
    "task1": ("accuracy", 0.92, "Accuracy > 92%"),
    "task2": ("MAE",      1.0,  "MAE < 1m"),
    "task3": ("accuracy", 0.90, "Accuracy > 90%"),
    "task4": ("f1",       0.80, "F1 > 0.80"),
}

for task in TASKS:
    results = load_results(task)
    metric_key, threshold, obj_str = objectives[task]

    if results:
        tm = results.get("test_metrics", {})
        val = tm.get(metric_key, tm.get("MAE", "N/A"))
        if isinstance(val, float):
            if task == "task2":
                status = "✅" if val < threshold else "❌"
            else:
                status = "✅" if val > threshold else "❌"
            val_str = f"{val:.4f}"
        else:
            val_str = str(val)
            status  = "?"
    else:
        val_str = "—"
        status  = "⏳"

    table_data.append([TASK_NAMES[task], metric_key, val_str, obj_str, status])

table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                  cellLoc="center", loc="center",
                  colColours=["#2c3e50"]*5)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Colorer l'en-tête
for j in range(5):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

fig2.suptitle("Tableau des Résultats — Skipper NDT", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "results_table.png", dpi=120, bbox_inches="tight")
print(f"Sauvegardé → {OUT_DIR}/results_table.png")

plt.show()
print("✅ Visualisation terminée !")
