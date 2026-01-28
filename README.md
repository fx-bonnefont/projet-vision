# Projet Vision - Segmentation Binaire
Ce projet permet d'entraîner un modèle de segmentation binaire (objets vs fond) sur des images aériennes/satellites (format DOTA).

Il utilise des **modèles de fondation** pré-entraînés (DINOv3, SAM3, etc.) pour des performances élevées avec peu de données.

## 📦 Installation

```bash
uv sync
```

## 📂 Structure des Données (Important)

Vos données doivent être organisées strictement comme suit :

```text
MON_DOSSIER_DOTA/
├── images/
│   ├── train/  (Images d'entraînement)
│   └── test/   (Images de validation/test)
├── labels/
│   ├── train/  (Labels d'entraînement .txt)
│   └── test/   (Labels de validation .txt)
└── debug/      (Facultatif, pour les visualisations)
```

## 🚀 Utilisation

### 1. Entraînement

Il suffit d'indiquer le dossier racine `--data`. Le script trouvera automatiquement les dossiers `train` et `test`.

```bash
uv run python train.py --data /chemin/vers/MON_DOSSIER_DOTA
```

*Options utiles :*
*   `--backbone dinov3_vit7b16_sat` (par défaut)
*   `--epochs 20`
*   `--batch-size 4`

### 2. Inférence (Test)

Pour tester le modèle (par défaut sur le dossier `images/test`) :

```bash
uv run python inference.py \
    --model model.pth \
    --data /chemin/vers/MON_DOSSIER_DOTA
```

*Le script détecte automatiquement le backbone utilisé lors de l'entraînement.*

## 🧠 Modèles Disponibles

*   **DINOv3** (Défaut) : `dinov3_vit7b16_sat` (Optimisé Satellite), `dinov3_vitb16`...
*   **SAM3** : `sam3`
*   **DINOv2** : `dinov2_vits14`...
*   **ResNet** : `resnet50`
