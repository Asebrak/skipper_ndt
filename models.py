"""
ÉTAPE 3 - MODÈLES CNN
Skipper NDT x HETIC

Architectures disponibles :
  - MagneticCNN    : CNN léger from scratch (4 canaux)
  - ResNet4ch      : ResNet18/34 adapté 4 canaux (Transfer Learning)

Supporte les 4 tâches via le paramètre `mode`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ─────────────────────────────────────────────────────────────────────────────
# 1. CNN LÉGER FROM SCRATCH
# ─────────────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Bloc Conv → BN → ReLU → MaxPool."""
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MagneticCNN(nn.Module):
    """
    CNN from scratch pour données magnétiques 4 canaux.

    mode="classify" → sortie (batch, num_classes) pour tâches 1, 3, 4
    mode="regress"  → sortie (batch,)             pour tâche 2 (width_m)
    """
    def __init__(self,
                 in_channels: int = 4,
                 num_classes: int = 2,
                 mode:        str = "classify",
                 dropout:     float = 0.4):
        super().__init__()
        self.mode = mode

        # Backbone
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),   # (4, 224, 224) → (32, 112, 112)
            ConvBlock(32, 64),            # → (64, 56, 56)
            ConvBlock(64, 128),           # → (128, 28, 28)
            ConvBlock(128, 256),          # → (256, 14, 14)
            ConvBlock(256, 512),          # → (512, 7, 7)
        )

        # Global Average Pooling → taille invariante
        self.gap = nn.AdaptiveAvgPool2d(1)    # (512, 1, 1)

        # Tête de prédiction
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
        )

        if mode == "classify":
            self.out = nn.Linear(64, num_classes)
        else:  # regress
            self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.head(x)
        out = self.out(x)
        if self.mode == "regress":
            return out.squeeze(1)           # (batch,)
        return out                          # (batch, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# 2. RESNET ADAPTÉ 4 CANAUX (TRANSFER LEARNING)
# ─────────────────────────────────────────────────────────────────────────────
class ResNet4ch(nn.Module):
    """
    ResNet18 ou ResNet34 pré-entraîné, adapté pour 4 canaux magnétiques.

    Stratégie : adapter la première couche conv en moyennant les poids RGB
    et en ajoutant un 4ème canal initialisé avec la moyenne.

    mode="classify" → tâches 1, 3, 4
    mode="regress"  → tâche 2
    """
    def __init__(self,
                 arch:        str   = "resnet18",
                 in_channels: int   = 4,
                 num_classes: int   = 2,
                 mode:        str   = "classify",
                 pretrained:  bool  = True,
                 dropout:     float = 0.3):
        super().__init__()
        self.mode = mode

        # Charger le backbone
        weights = "IMAGENET1K_V1" if pretrained else None
        if arch == "resnet18":
            backbone = models.resnet18(weights=weights)
        elif arch == "resnet34":
            backbone = models.resnet34(weights=weights)
        elif arch == "resnet50":
            backbone = models.resnet50(weights=weights)
        else:
            raise ValueError(f"arch inconnu : {arch}")

        # Adapter la couche d'entrée : 3 → 4 canaux
        old_conv    = backbone.conv1
        new_conv    = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        # Initialiser avec les poids pré-entraînés (3 canaux) + 4ème = moyenne
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3]  = old_conv.weight.mean(dim=1)
        backbone.conv1 = new_conv

        # Extraire le backbone (sans la couche FC finale)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → (batch, 512/2048, 1, 1)
        feat_dim = backbone.fc.in_features   # 512 pour R18/34, 2048 pour R50

        # Nouvelle tête
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
        )

        if mode == "classify":
            self.out = nn.Linear(256, num_classes)
        else:
            self.out = nn.Linear(256, 1)

    def forward(self, x):
        x   = self.backbone(x)
        x   = self.classifier(x)
        out = self.out(x)
        if self.mode == "regress":
            return out.squeeze(1)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. FACTORY : CRÉER LE BON MODÈLE SELON LA TÂCHE
# ─────────────────────────────────────────────────────────────────────────────
def build_model(task:       str = "task1",
                arch:       str = "resnet18",
                pretrained: bool = True,
                dropout:    float = 0.3) -> nn.Module:
    """
    Crée et retourne le modèle approprié pour la tâche demandée.

    task1 → classify, 2 classes (absent / présent)
    task2 → regress  (width_m)
    task3 → classify, 2 classes (insuffisant / suffisant)
    task4 → classify, 2 classes (single / parallel)
    """
    if task == "task2":
        mode        = "regress"
        num_classes = 1
    else:
        mode        = "classify"
        num_classes = 2

    if arch == "scratch":
        model = MagneticCNN(in_channels=4, num_classes=num_classes,
                            mode=mode, dropout=dropout)
        print(f"Modèle : MagneticCNN (from scratch) — {task} [{mode}]")
    else:
        model = ResNet4ch(arch=arch, in_channels=4, num_classes=num_classes,
                          mode=mode, pretrained=pretrained, dropout=dropout)
        print(f"Modèle : ResNet4ch ({arch}, pretrained={pretrained}) — {task} [{mode}]")

    # Afficher le nombre de paramètres
    total  = sum(p.numel() for p in model.parameters())
    trainb = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Paramètres totaux : {total:,} | Entraînables : {trainb:,}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# TEST RAPIDE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("TEST DES ARCHITECTURES")
    print("=" * 60)

    dummy = torch.randn(4, 4, 224, 224)  # batch=4, channels=4, 224x224

    for task in ["task1", "task2", "task3", "task4"]:
        print(f"\n── {task.upper()} ──")
        for arch in ["scratch", "resnet18"]:
            model  = build_model(task=task, arch=arch, pretrained=False)
            output = model(dummy)
            print(f"  Output shape ({arch}) : {output.shape}")

    print("\n✅ Architectures OK !")
