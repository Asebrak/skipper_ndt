"""
ÉTAPE 2 - DATASET & PREPROCESSING
Skipper NDT x HETIC

Ce fichier définit :
  - MagneticDataset : Dataset PyTorch générique (4 canaux)
  - get_dataloaders : Fonction de création des DataLoaders (train/val/test)
  - Preprocessing : normalisation par canal + resize adaptif
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Literal


# ── CONSTANTES ───────────────────────────────────────────────────────────────
CHANNELS   = ["Bx", "By", "Bz", "Norm"]
TARGET_IMG = 224   # taille cible pour le resize (carré)

# Statistiques précalculées sur le dataset (à affiner après exploration)
# Si vous voulez les calculer vous-même, lancez compute_stats()
CHANNEL_MEANS = None   # sera calculé dynamiquement si None
CHANNEL_STDS  = None


# ── FONCTION : CHARGER UN FICHIER NPZ ────────────────────────────────────────
def load_field(path: Path) -> np.ndarray:
    """
    Charge un fichier .npz et retourne un array (4, H, W) float32.
    Gère les différents formats possibles.
    """
    with np.load(path) as npz:
        keys = list(npz.keys())
        data = npz[keys[0]].astype(np.float32)

    # Normaliser la shape → (C, H, W)
    if data.ndim == 2:
        # Cas 1 seul canal
        data = data[np.newaxis]                      # (1, H, W)
        data = np.repeat(data, 4, axis=0)            # (4, H, W) dupliqué

    elif data.ndim == 3:
        if data.shape[0] == 4:
            pass                                     # déjà (4, H, W)
        elif data.shape[2] == 4:
            data = data.transpose(2, 0, 1)           # (H, W, 4) → (4, H, W)
        elif data.shape[0] not in [4]:
            # Prendre les 4 premiers ou dupliquer
            data = data[:4] if data.shape[0] >= 4 else np.repeat(data[[0]], 4, axis=0)

    assert data.shape[0] == 4, f"Attendu 4 canaux, reçu {data.shape[0]}"
    return data   # (4, H, W)


# ── PREPROCESSING : RESIZE ADAPTATIF ─────────────────────────────────────────
def adaptive_resize(data: np.ndarray, target: int = TARGET_IMG) -> np.ndarray:
    """
    Resize une image (4, H, W) vers (4, target, target).
    Stratégie : pad pour rendre carré, puis resize.
    Préserve les ratios physiques.
    """
    import cv2

    C, H, W = data.shape
    # Padding pour rendre carré
    max_dim = max(H, W)
    pad_h   = (max_dim - H) // 2
    pad_w   = (max_dim - W) // 2

    padded = np.pad(data, ((0, 0), (pad_h, max_dim - H - pad_h),
                           (pad_w, max_dim - W - pad_w)),
                    mode="constant", constant_values=0)  # (4, max_dim, max_dim)

    # Resize canal par canal
    resized = np.stack([
        cv2.resize(padded[c], (target, target), interpolation=cv2.INTER_LINEAR)
        for c in range(C)
    ])  # (4, target, target)

    return resized


# ── PREPROCESSING : NORMALISATION ────────────────────────────────────────────
def normalize_channels(data: np.ndarray,
                        means: Optional[np.ndarray] = None,
                        stds:  Optional[np.ndarray] = None) -> np.ndarray:
    """
    Normalise chaque canal indépendamment.
    Si means/stds non fournis → normalise par les stats locales de l'image.
    """
    if means is not None and stds is not None:
        for c in range(data.shape[0]):
            data[c] = (data[c] - means[c]) / (stds[c] + 1e-8)
    else:
        for c in range(data.shape[0]):
            mu  = data[c].mean()
            std = data[c].std() + 1e-8
            data[c] = (data[c] - mu) / std
    return data


# ── DATASET PYTORCH ───────────────────────────────────────────────────────────
class MagneticDataset(Dataset):
    """
    Dataset PyTorch pour les données magnétiques Skipper NDT.

    Paramètres
    ----------
    df         : DataFrame avec les métadonnées (issu du CSV)
    data_dir   : Dossier contenant les fichiers .npz
    task       : "task1" | "task2" | "task3" | "task4"
    target_size: Taille carrée cible après resize
    augment    : Appliquer data augmentation (train uniquement)
    means/stds : Stats de normalisation (None = normalisation locale)
    """

    TASK_LABEL_MAP = {
        "task1": "label",           # 0/1 binaire
        "task2": "width_m",         # flottant
        "task3": "coverage_type",   # catégoriel → encodé
        "task4": "pipe_type",       # "single"=0 / "parallel"=1
    }

    def __init__(self,
                 df:          pd.DataFrame,
                 data_dir:    Path,
                 task:        Literal["task1","task2","task3","task4"] = "task1",
                 target_size: int  = TARGET_IMG,
                 augment:     bool = False,
                 means:       Optional[np.ndarray] = None,
                 stds:        Optional[np.ndarray]  = None):

        self.df          = df.reset_index(drop=True)
        self.data_dir    = Path(data_dir)
        self.task        = task
        self.target_size = target_size
        self.augment     = augment
        self.means       = means
        self.stds        = stds

        # Encoder les labels selon la tâche
        self._prepare_labels()

    def _prepare_labels(self):
        col = self.TASK_LABEL_MAP[self.task]

        if self.task == "task1":
            self.labels = self.df["label"].values.astype(np.int64)

        elif self.task == "task2":
            self.labels = self.df["width_m"].values.astype(np.float32)

        elif self.task == "task3":
            # "perfect"=1 / autres=0 (courant suffisant vs insuffisant)
            self.labels = (self.df["coverage_type"] == "perfect").astype(np.int64).values

        elif self.task == "task4":
            self.labels = (self.df["pipe_type"] == "parallel").astype(np.int64).values

    def _augment(self, data: np.ndarray) -> np.ndarray:
        """Augmentations préservant la physique magnétique."""
        # Flip horizontal
        if np.random.rand() > 0.5:
            data = data[:, :, ::-1].copy()
        # Flip vertical
        if np.random.rand() > 0.5:
            data = data[:, ::-1, :].copy()
        # Rotation 90° (magnétiquement cohérente)
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            data = np.rot90(data, k=k, axes=(1, 2)).copy()
        # Ajout de bruit gaussien léger
        if np.random.rand() > 0.7:
            noise = np.random.normal(0, 0.02, data.shape).astype(np.float32)
            data  = data + noise
        return data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = self.df.loc[idx, "field_file"]
        path  = self.data_dir / fname

        # 1. Charger
        data = load_field(path)                        # (4, H, W)

        # 2. Resize adaptatif
        data = adaptive_resize(data, self.target_size) # (4, T, T)

        # 3. Normalisation
        data = normalize_channels(data, self.means, self.stds)

        # 4. Augmentation (train uniquement)
        if self.augment:
            data = self._augment(data)

        # 5. Tensor
        tensor = torch.from_numpy(data.copy())         # (4, T, T)

        label = self.labels[idx]
        if self.task == "task2":
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = torch.tensor(label, dtype=torch.long)

        return tensor, label


# ── SPLIT & DATALOADERS ───────────────────────────────────────────────────────
def get_dataloaders(csv_path:    Path,
                    data_dir:    Path,
                    task:        str  = "task1",
                    target_size: int  = TARGET_IMG,
                    batch_size:  int  = 32,
                    val_ratio:   float= 0.15,
                    test_ratio:  float= 0.15,
                    seed:        int  = 42,
                    num_workers: int  = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Retourne (train_loader, val_loader, test_loader).
    Split stratifié sur le label binaire (tâche 1, 3, 4) ou simple (tâche 2).
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path, sep=";")

    # Pour task4 : filtrer uniquement les échantillons avec conduite
    if task == "task4":
        df = df[df["label"] == 1].reset_index(drop=True)
        stratify_col = (df["pipe_type"] == "parallel").astype(int)
    elif task == "task2":
        df = df[df["label"] == 1].reset_index(drop=True)
        stratify_col = None
    elif task == "task3":
        stratify_col = (df["coverage_type"] == "perfect").astype(int)
    else:
        stratify_col = df["label"]

    # Split train / temp
    idx = np.arange(len(df))
    idx_train, idx_temp = train_test_split(
        idx, test_size=(val_ratio + test_ratio),
        stratify=stratify_col.iloc[idx] if stratify_col is not None else None,
        random_state=seed
    )
    # Split val / test
    strat_temp = stratify_col.iloc[idx_temp] if stratify_col is not None else None
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=strat_temp,
        random_state=seed
    )

    df_train = df.iloc[idx_train]
    df_val   = df.iloc[idx_val]
    df_test  = df.iloc[idx_test]

    print(f"Split [{task}] → train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    train_ds = MagneticDataset(df_train, data_dir, task, target_size, augment=True)
    val_ds   = MagneticDataset(df_val,   data_dir, task, target_size, augment=False)
    test_ds  = MagneticDataset(df_test,  data_dir, task, target_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# ── CALCUL DES STATS DE NORMALISATION ────────────────────────────────────────
def compute_stats(csv_path: Path, data_dir: Path,
                  n_samples: int = 200, target_size: int = TARGET_IMG):
    """
    Calcule les moyennes et écarts-types par canal sur n_samples fichiers.
    Utile pour une normalisation globale cohérente.
    """
    df = pd.read_csv(csv_path, sep=";")
    df = df.sample(min(n_samples, len(df)), random_state=42)

    sums   = np.zeros(4, dtype=np.float64)
    sq_sum = np.zeros(4, dtype=np.float64)
    count  = 0

    for fname in df["field_file"]:
        path = data_dir / fname
        data = load_field(path)
        data = adaptive_resize(data, target_size)
        sums   += data.reshape(4, -1).sum(axis=1)
        sq_sum += (data.reshape(4, -1) ** 2).sum(axis=1)
        count  += data.shape[1] * data.shape[2]

    means = (sums / count).astype(np.float32)
    stds  = np.sqrt(sq_sum / count - means**2).astype(np.float32)

    print("Moyennes par canal :", means)
    print("Écarts-types       :", stds)
    return means, stds


if __name__ == "__main__":
    # Test rapide
    DATA_DIR = Path("Training_database_float16")
    CSV_PATH = Path("pipe_detection_label.csv")

    print("Calcul des statistiques de normalisation...")
    means, stds = compute_stats(CSV_PATH, DATA_DIR, n_samples=100)

    print("\nCréation des DataLoaders (task1)...")
    train_l, val_l, test_l = get_dataloaders(CSV_PATH, DATA_DIR, task="task1")

    batch, labels = next(iter(train_l))
    print(f"\nBatch shape : {batch.shape}")     # (32, 4, 224, 224)
    print(f"Labels shape: {labels.shape}")      # (32,)
    print(f"Dtype       : {batch.dtype}")
    print(f"Min/Max     : {batch.min():.3f} / {batch.max():.3f}")
    print("\n✅ Dataset OK !")
