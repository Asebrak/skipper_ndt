# 🔍 Skipper NDT x HETIC — Identification de Pipes par ML

## Structure du projet

```
skipper_project/
├── 01_explore_data.py   # ÉTAPE 1 : Exploration et visualisation des données
├── dataset.py           # ÉTAPE 2 : Dataset PyTorch + preprocessing
├── models.py            # ÉTAPE 3 : Architectures CNN (scratch & ResNet)
├── train.py             # ÉTAPE 4 : Entraînement des 4 tâches
├── inference.py         # ÉTAPE 5 : Inférence (script obligatoire Skipper)
├── plot_results.py      # ÉTAPE 6 : Visualisation des résultats
├── models/              # Modèles sauvegardés après entraînement
│   ├── task1/best_model.pth
│   ├── task2/best_model.pth
│   ├── task3/best_model.pth
│   └── task4/best_model.pth
└── exploration_outputs/ # Graphiques générés
```

---

## 📦 Installation

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn opencv-python tqdm
```

> **Mac avec Apple Silicon (M1/M2/M3)** : PyTorch supporte MPS nativement, l'entraînement sera accéléré automatiquement.

---

## 🚀 Workflow étape par étape

### 1. Placer les fichiers de données

```
Training_database_float16/   ← dossier avec les 2833 fichiers .npz
pipe_detection_label.csv     ← fichier CSV de labels
skipper_project/             ← ce dossier
```

### 2. Explorer les données

```bash
cd chemin/vers/dossier/contenant/Training_database_float16/
python skipper_project/01_explore_data.py
```
→ Génère des visualisations dans `exploration_outputs/`

### 3. Entraîner les modèles

```bash
# Tâche 1 : Présence de conduite (binaire)
python skipper_project/train.py --task task1 --arch resnet18 --epochs 30

# Tâche 2 : Largeur magnétique (régression)
python skipper_project/train.py --task task2 --arch resnet18 --epochs 40

# Tâche 3 : Intensité du courant (binaire)
python skipper_project/train.py --task task3 --arch resnet18 --epochs 25

# Tâche 4 : Conduites parallèles (binaire, dataset plus petit)
python skipper_project/train.py --task task4 --arch resnet18 --epochs 35 --dropout 0.5
```

> **Astuce** : Pour des tests rapides, utiliser `--arch scratch` (pas de téléchargement de poids).

### 4. Inférence (livrable obligatoire)

```bash
# Sur un fichier unique — toutes les tâches
python skipper_project/inference.py --all --input sample_00000_perfect_straight_clean_field.npz

# Sur une tâche spécifique
python skipper_project/inference.py --task task1 --input sample_00000.npz

# Sur un dossier entier → export CSV
python skipper_project/inference.py --all --input_dir Training_database_float16/ --output predictions.csv
```

### 5. Visualiser les résultats

```bash
python skipper_project/plot_results.py
```

---

## 🎯 Objectifs par tâche

| Tâche | Type | Objectif | Métrique |
|-------|------|----------|---------|
| T1 : Présence de conduite | Classification | Accuracy > 92%, Recall > 95% | CrossEntropy |
| T2 : Largeur magnétique | Régression | MAE < 1m | SmoothL1 |
| T3 : Intensité courant | Classification | Accuracy > 90%, Recall > 85% | CrossEntropy |
| T4 : Conduites parallèles | Classification | F1 > 0.80 | CrossEntropy |

---

## 📁 Livrables à soumettre

Pour chaque tâche, créer un dossier :

```
task1/
├── best_model.pth   ← modèle PyTorch entraîné
├── train.py         ← script d'entraînement (optionnel)
└── inference.py     ← script d'inférence (OBLIGATOIRE)
```

**Usage de inference.py (attendu par Skipper) :**
```bash
python inference.py --task task1 --input chemin/vers/image.npz
```

---

## 💡 Notes techniques

### Format des fichiers NPZ
- Chaque fichier contient les 4 canaux magnétiques : **Bx, By, Bz, Norm**
- Unité : nanoTesla (nT)
- Dimensions variables : 150×150 à 4000×3750 pixels
- 1 pixel = 0.2m × 0.2m

### Preprocessing
- **Padding + Resize** vers 224×224 (préserve les ratios)
- **Normalisation** par canal (μ=0, σ=1)
- **Augmentation** : flip H/V, rotation 90°, bruit gaussien

### Architecture
- **ResNet18 adapté 4 canaux** : 1ère couche modifiée pour accepter 4 canaux
- **Global Average Pooling** : invariant à la taille d'entrée
- **Transfer Learning** depuis ImageNet + fine-tuning complet

### Labels utilisés
- T1 : colonne `label` (0=absent, 1=présent)
- T2 : colonne `width_m` (flottant, 2-155m)
- T3 : colonne `coverage_type` ("perfect"=1, autres=0)
- T4 : colonne `pipe_type` ("parallel"=1, "single"=0)
