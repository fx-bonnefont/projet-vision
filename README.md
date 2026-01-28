# Projet Vision - Segmentation Binaire

Ce projet permet d'entraîner un modèle de segmentation binaire (objets vs fond) sur des images aériennes/satellites (format DOTA).

Il utilise des **modèles de fondation (Backbones)** pré-entraînés (DINOv2, DINOv3, SAM3) pour extraire des features puissantes sans avoir besoin d'un gros dataset.

## Installation

```bash
uv sync
```

## Utilisation

### 1. Entraînement

Pour entraîner le modèle, il faut spécifier les dossiers d'images et de labels.
Vous pouvez optionnellement ajouter un jeu de validation pour surveiller le sur-apprentissage.

```bash
uv run python train.py \
    --backbone dinov3_vitb16 \
    --images ./train/images \
    --labels ./train/labels \
    --val-images ./test/images \
    --val-labels ./test/labels \
    --epochs 20
```

### 2. Inférence (Test)

Pour tester le modèle sur de nouvelles images et visualiser le résultat :

```bash
uv run python inference.py \
    --model model.pth \
    --images ./test/images
```
*Le script détecte automatiquement quel backbone a été utilisé lors de l'entraînement.*

## Modèles Disponibles

Vous pouvez choisir le backbone avec l'argument `--backbone` :

*   **DINOv3** (recommandé) : `dinov3_vits16`, `dinov3_vitb16`, `dinov3_vitl16`, `dinov3_vit7b16_sat` (spécial satellite).
*   **DINOv2** : `dinov2_vits14`, `dinov2_vitb14`...
*   **SAM3** : `sam3`
*   **ResNet** (classique) : `resnet50`
