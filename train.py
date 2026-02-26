"""
ÉTAPE 4 - ENTRAÎNEMENT
Skipper NDT x HETIC

Usage :
  python train.py --task task1 --arch resnet18 --epochs 30
  python train.py --task task2 --arch resnet18 --epochs 40
  python train.py --task task3 --arch scratch  --epochs 25
  python train.py --task task4 --arch resnet18 --epochs 30

Le modèle entraîné est sauvegardé dans : models/{task}/best_model.pth
"""

import argparse
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (accuracy_score, recall_score,
                             f1_score, mean_absolute_error)

# Imports locaux
from dataset import get_dataloaders
from models  import build_model


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG PAR DÉFAUT
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "data_dir"   : "Training_database_float16",
    "csv_path"   : "pipe_detection_label.csv",
    "models_dir" : "models",
    "target_size": 224,
    "batch_size" : 32,
    "num_workers": 4,
    "lr"         : 1e-4,
    "weight_decay": 1e-4,
    "dropout"    : 0.3,
}


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device : GPU — {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device : Apple MPS (GPU Mac)")
    else:
        device = torch.device("cpu")
        print("Device : CPU")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_loss(task: str, train_loader=None, device=None):
    """Retourne la fonction de perte adaptée à la tâche."""
    if task == "task2":
        return nn.SmoothL1Loss()   # Huber loss → robuste aux outliers

    # Pour les tâches de classification, calculer les poids de classes
    if train_loader is not None:
        labels = []
        for _, y in train_loader:
            labels.extend(y.numpy())
        labels = np.array(labels)
        counts = np.bincount(labels)
        weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)
        if device:
            weights = weights.to(device)
        print(f"  Poids de classes : {weights.numpy().round(3)}")
        return nn.CrossEntropyLoss(weight=weights)

    return nn.CrossEntropyLoss()


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(task: str, all_preds, all_labels, all_probs=None):
    """Calcule les métriques selon la tâche."""
    if task == "task2":
        mae = mean_absolute_error(all_labels, all_preds)
        return {"MAE": round(mae, 4), "objectif": "MAE < 1.0m"}

    acc    = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1     = f1_score(all_labels, all_preds, zero_division=0)

    metrics = {"accuracy": round(acc, 4),
               "recall"  : round(recall, 4),
               "f1"      : round(f1, 4)}

    if task == "task1":
        metrics["objectif"] = f"Acc>0.92: {'✅' if acc>0.92 else '❌'} | Recall>0.95: {'✅' if recall>0.95 else '❌'}"
    elif task == "task3":
        metrics["objectif"] = f"Acc>0.90: {'✅' if acc>0.90 else '❌'} | Recall>0.85: {'✅' if recall>0.85 else '❌'}"
    elif task == "task4":
        metrics["objectif"] = f"F1>0.80: {'✅' if f1>0.80 else '❌'}"

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# BOUCLE D'ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, task, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)

        if task == "task2":
            loss   = criterion(output, y)
            preds  = output.detach().cpu().numpy()
            labels = y.cpu().numpy()
        else:
            loss   = criterion(output, y)
            preds  = output.argmax(dim=1).cpu().numpy()
            labels = y.cpu().numpy()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(preds)
        all_labels.extend(labels)

        if (batch_idx + 1) % 10 == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] loss={loss.item():.4f}", end="\r")

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(task, np.array(all_preds), np.array(all_labels))
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, task, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)

        if task == "task2":
            loss   = criterion(output, y)
            preds  = output.cpu().numpy()
            labels = y.cpu().numpy()
        else:
            loss   = criterion(output, y)
            preds  = output.argmax(dim=1).cpu().numpy()
            labels = y.cpu().numpy()

        total_loss += loss.item()
        all_preds.extend(preds)
        all_labels.extend(labels)

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(task, np.array(all_preds), np.array(all_labels))
    return avg_loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
# ENTRAÎNEMENT COMPLET
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    device = get_device()

    # Dossier de sauvegarde
    save_dir = Path(args.models_dir) / args.task
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT — {args.task.upper()} | arch={args.arch}")
    print(f"{'='*60}\n")

    # DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        csv_path    = Path(args.csv_path),
        data_dir    = Path(args.data_dir),
        task        = args.task,
        target_size = args.target_size,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    # Modèle
    model = build_model(task=args.task, arch=args.arch,
                        pretrained=args.pretrained, dropout=args.dropout)
    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = get_loss(args.task, train_loader, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Métrique de suivi (pour sauvegarder le meilleur modèle)
    best_score = float("inf") if args.task == "task2" else 0.0
    history    = {"train_loss": [], "val_loss": [], "train_metrics": [], "val_metrics": []}

    print(f"\nEntraînement sur {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, args.task, device)
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, args.task, device)

        scheduler.step()

        # Affichage
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")
        print(f"  Train: {train_metrics}")
        print(f"  Val  : {val_metrics}")

        # Sauvegarder le meilleur modèle
        if args.task == "task2":
            score = val_metrics["MAE"]
            is_best = score < best_score
        elif args.task == "task4":
            score   = val_metrics["f1"]
            is_best = score > best_score
        else:
            score   = val_metrics["accuracy"]
            is_best = score > best_score

        if is_best:
            best_score = score
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "val_metrics" : val_metrics,
                "args"        : vars(args),
            }, save_dir / "best_model.pth")
            print(f"  ✅ Meilleur modèle sauvegardé ! score={best_score:.4f}")

        # Historique
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_metrics"].append(train_metrics)
        history["val_metrics"].append(val_metrics)
        print()

    # ── Évaluation finale sur le test set ────────────────────────────────────
    print("=" * 60)
    print("ÉVALUATION FINALE SUR LE TEST SET")
    print("=" * 60)

    # Charger le meilleur modèle
    checkpoint = torch.load(save_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    _, test_metrics = evaluate(model, test_loader, criterion, args.task, device)
    print(f"Test metrics : {test_metrics}")

    # Sauvegarder l'historique
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Sauvegarder les métriques finales
    results = {"task": args.task, "arch": args.arch,
               "best_val_score": best_score, "test_metrics": test_metrics}
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Entraînement terminé ! Modèle sauvegardé dans : {save_dir}/")
    return model, test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement Skipper NDT")

    parser.add_argument("--task",        type=str,   default="task1",
                        choices=["task1","task2","task3","task4"])
    parser.add_argument("--arch",        type=str,   default="resnet18",
                        choices=["scratch","resnet18","resnet34","resnet50"])
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",  type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",          type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight_decay",type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--dropout",     type=float, default=DEFAULT_CONFIG["dropout"])
    parser.add_argument("--target_size", type=int,   default=DEFAULT_CONFIG["target_size"])
    parser.add_argument("--num_workers", type=int,   default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--data_dir",    type=str,   default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--csv_path",    type=str,   default=DEFAULT_CONFIG["csv_path"])
    parser.add_argument("--models_dir",  type=str,   default=DEFAULT_CONFIG["models_dir"])
    parser.add_argument("--pretrained",  action="store_true", default=True)
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")

    args = parser.parse_args()
    train(args)
