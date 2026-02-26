"""
INFÉRENCE — Skipper NDT x HETIC
================================
Script d'inférence obligatoire selon le cahier des charges.

Usage :
  # Prédire une seule tâche sur un fichier NPZ
  python inference.py --task task1 --input path/to/sample.npz

  # Prédire toutes les tâches d'un coup
  python inference.py --all --input path/to/sample.npz

  # Inférence sur un dossier entier (exporte en CSV)
  python inference.py --all --input_dir path/to/folder/ --output results.csv

Pré-requis :
  - Les modèles doivent être entraînés et se trouver dans models/{task}/best_model.pth
  - Ce fichier doit être dans le même dossier que les modèles (ou ajuster MODEL_BASE_DIR)
"""

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Imports locaux
from dataset import load_field, adaptive_resize, normalize_channels
from models  import build_model

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_BASE_DIR = Path("models")   # dossier contenant task1/, task2/, task3/, task4/
TARGET_SIZE    = 224
TASKS          = ["task1", "task2", "task3", "task4"]

TASK_INFO = {
    "task1": {"mode": "classify", "label_map": {0: "Pas de conduite", 1: "Conduite présente"}},
    "task2": {"mode": "regress",  "label_map": None},
    "task3": {"mode": "classify", "label_map": {0: "Courant insuffisant", 1: "Courant suffisant"}},
    "task4": {"mode": "classify", "label_map": {0: "Conduite unique", 1: "Conduites parallèles"}},
}


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEUR DE MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
class PipelinePredictor:
    """
    Prédicateur multi-tâches pour les données magnétiques Skipper NDT.
    Charge les modèles à la demande (lazy loading).
    """

    def __init__(self, model_base_dir: Path = MODEL_BASE_DIR, device: str = "auto"):
        self.base_dir = Path(model_base_dir)
        self._models  = {}

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"[Inférence] Device : {self.device}")

    def _load_model(self, task: str) -> torch.nn.Module:
        """Charge (ou retourne depuis le cache) le modèle pour une tâche."""
        if task in self._models:
            return self._models[task]

        ckpt_path = self.base_dir / task / "best_model.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {ckpt_path}\n"
                f"Lancez d'abord : python train.py --task {task}"
            )

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        args       = checkpoint.get("args", {})
        arch       = args.get("arch", "resnet18")

        model = build_model(task=task, arch=arch, pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)
        model.eval()

        val_metrics = checkpoint.get("val_metrics", {})
        print(f"[{task}] Modèle chargé ({arch}) | val_metrics={val_metrics}")

        self._models[task] = model
        return model

    def preprocess(self, npz_path: Path) -> torch.Tensor:
        """Charge et prépare un fichier NPZ pour l'inférence."""
        data = load_field(npz_path)                         # (4, H, W)
        data = adaptive_resize(data, TARGET_SIZE)           # (4, 224, 224)
        data = normalize_channels(data)                     # normalisation locale
        tensor = torch.from_numpy(data.copy()).unsqueeze(0) # (1, 4, 224, 224)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict_task(self, npz_path: Path, task: str) -> dict:
        """Prédit pour une tâche sur un fichier NPZ."""
        model  = self._load_model(task)
        tensor = self.preprocess(npz_path)
        output = model(tensor)

        info = TASK_INFO[task]

        if info["mode"] == "regress":
            value = output.item()
            return {
                "task"       : task,
                "prediction" : round(value, 2),
                "unit"       : "mètres",
                "interpretation": f"Largeur magnétique estimée : {value:.2f} m"
            }
        else:
            probs    = F.softmax(output, dim=1).squeeze().cpu().numpy()
            pred_cls = int(probs.argmax())
            conf     = float(probs[pred_cls])
            label    = info["label_map"][pred_cls]
            return {
                "task"          : task,
                "prediction"    : pred_cls,
                "label"         : label,
                "confidence"    : round(conf, 4),
                "probabilities" : {info["label_map"][i]: round(float(p), 4)
                                   for i, p in enumerate(probs)},
                "interpretation": f"{label} (confiance : {conf*100:.1f}%)"
            }

    def predict_all(self, npz_path: Path) -> dict:
        """Prédit pour les 4 tâches sur un fichier NPZ."""
        results = {"file": str(npz_path)}
        for task in TASKS:
            try:
                results[task] = self.predict_task(npz_path, task)
            except FileNotFoundError as e:
                results[task] = {"error": str(e)}
        return results


# ─────────────────────────────────────────────────────────────────────────────
# AFFICHAGE DES RÉSULTATS
# ─────────────────────────────────────────────────────────────────────────────
def print_results(results: dict):
    print("\n" + "=" * 60)
    print(f"RÉSULTATS — {results.get('file', '')}")
    print("=" * 60)

    task_names = {
        "task1": "Tâche 1 : Présence de conduite",
        "task2": "Tâche 2 : Largeur magnétique (Map Width)",
        "task3": "Tâche 3 : Intensité du courant",
        "task4": "Tâche 4 : Conduites parallèles",
    }

    for task in TASKS:
        if task not in results:
            continue
        r = results[task]
        print(f"\n── {task_names.get(task, task)} ──")
        if "error" in r:
            print(f"  ⚠️  {r['error']}")
        else:
            print(f"  {r.get('interpretation', r)}")
            if "probabilities" in r:
                for label, prob in r["probabilities"].items():
                    print(f"    {label}: {prob*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# INFÉRENCE SUR UN DOSSIER
# ─────────────────────────────────────────────────────────────────────────────
def predict_folder(predictor: PipelinePredictor, input_dir: Path, output_csv: Path):
    """Inférence sur tous les .npz d'un dossier, export CSV."""
    import pandas as pd

    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        print(f"Aucun fichier .npz trouvé dans : {input_dir}")
        return

    print(f"Inférence sur {len(npz_files)} fichiers...")
    rows = []

    for i, npz_path in enumerate(npz_files):
        print(f"  [{i+1}/{len(npz_files)}] {npz_path.name}", end="\r")
        results = predictor.predict_all(npz_path)
        row = {"file": npz_path.name}

        for task in TASKS:
            r = results.get(task, {})
            if "error" in r:
                row[f"{task}_pred"] = None
            elif task == "task2":
                row[f"{task}_width_m"] = r.get("prediction")
            else:
                row[f"{task}_pred"]  = r.get("prediction")
                row[f"{task}_label"] = r.get("label")
                row[f"{task}_conf"]  = r.get("confidence")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Résultats exportés → {output_csv}")
    print(df.head())


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inférence Skipper NDT")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",     type=str, help="Chemin vers un fichier .npz")
    group.add_argument("--input_dir", type=str, help="Dossier contenant des fichiers .npz")

    parser.add_argument("--task",       type=str, default=None,
                        choices=["task1","task2","task3","task4"],
                        help="Tâche spécifique (défaut : toutes)")
    parser.add_argument("--all",        action="store_true",
                        help="Prédire toutes les tâches")
    parser.add_argument("--output",     type=str, default="predictions.csv",
                        help="Fichier CSV de sortie (pour --input_dir)")
    parser.add_argument("--model_dir",  type=str, default="models")
    parser.add_argument("--device",     type=str, default="auto")

    args = parser.parse_args()

    predictor = PipelinePredictor(
        model_base_dir=Path(args.model_dir),
        device=args.device
    )

    if args.input_dir:
        # Mode batch
        predict_folder(predictor, Path(args.input_dir), Path(args.output))

    elif args.input:
        npz_path = Path(args.input)
        if not npz_path.exists():
            print(f"Fichier introuvable : {npz_path}")
            sys.exit(1)

        if args.task and not args.all:
            # Une seule tâche
            result = predictor.predict_task(npz_path, args.task)
            print_results({"file": str(npz_path), args.task: result})
        else:
            # Toutes les tâches
            results = predictor.predict_all(npz_path)
            print_results(results)
            print("\nJSON brut :")
            print(json.dumps(results, indent=2, ensure_ascii=False))
