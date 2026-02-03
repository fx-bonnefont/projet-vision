# Segmentation vit + conv1x1

Segmentation sémantique d'objets aériens avec des visions transformers et décodeurs modulaires.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Image RGB (512x512)                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │  DINOv3 Backbone │  (gelé, pré-entraîné)                 │
│  │  - small/base/large/huge/giant                           │
│  │  - large-sat/giant-sat (satellite)                       │
│  └────────┬────────┘                                        │
│           │ 4 feature maps multi-échelles                   │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │    Décodeur     │  (entraînable)                         │
│  │  - heavy_unet (~50M params)                              │
│  │  - medium (~10M params)                                  │
│  │  - light (~500K params)                                  │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  Masque de segmentation (512x512)                           │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prérequis

- Python >= 3.10
- GPU CUDA (recommandé) ou Apple Silicon (MPS)
- Token HuggingFace pour accéder aux modèles DINOv3

### Setup

```bash
# Cloner le repo
git clone <repo-url>
cd segdino

# Installer les dépendances avec uv
uv sync

# Configurer le token HuggingFace
echo "HF_TOKEN=hf_xxxxx" > .env
```

Le token HuggingFace est nécessaire car les modèles DINOv3 sont en accès restreint. Demandez l'accès sur [HuggingFace](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m).

## Structure des données

Le dataset doit être organisé comme suit :

```
segdata/DOTA/DOTA_PLANES_TILED/
├── train/
│   ├── image/
│   │   ├── tile_0001.png
│   │   └── ...
│   └── mask/
│       ├── tile_0001.png   (même nom que l'image)
│       └── ...
└── test/
    ├── image/
    └── mask/
```

- Images : RGB, 512x512 pixels, format PNG
- Masques : Grayscale, 512x512 pixels, format PNG (blanc = objet, noir = fond)

## Entraînement

### Commande de base

```bash
uv run python train.py \
    --data_dir segdata/DOTA/DOTA_PLANES_TILED \
    --model_size large-sat \
    --decoder heavy_unet \
    --target_type mask \
    --loss center \
    --epochs 10 \
    --batch_size 8 \
    --lr 5e-4
```

### Options principales

| Argument | Valeurs | Description |
|----------|---------|-------------|
| `--model_size` | `small`, `small-plus`, `base`, `large`, `huge`, `giant`, `large-sat`, `giant-sat` | Taille du backbone DINOv3 |
| `--decoder` | `heavy_unet`, `medium`, `light` | Architecture du décodeur |
| `--target_type` | `mask`, `center` | Mode segmentation ou détection de centres |
| `--loss` | `combo`, `dice`, `mse`, `focal`, `center` | Fonction de perte |
| `--sigma` | float (défaut: 8.0) | Sigma des gaussiennes (mode center) |
| `--batch_size` | int | Taille de batch par GPU |
| `--lr` | float | Learning rate (défaut: 5e-4) |

### Backbones recommandés

| Backbone | Usage | VRAM |
|----------|-------|------|
| `large-sat` | **Production** - Pré-entraîné sur images satellite | ~16 GB |
| `base` | Développement rapide | ~8 GB |
| `small-plus` | Ressources limitées | ~4 GB |

### Sorties

Les checkpoints et logs sont sauvegardés dans `runs/` :

```
runs/
├── 20250203_12_heavy_unet_mask_a1b2_best.pth   # Meilleur modèle
└── 20250203_12_heavy_unet_mask_a1b2_log.csv    # Logs d'entraînement
```

## Inférence

### Sur des images complètes

```bash
uv run python inference.py \
    --ckpt runs/20250203_12_heavy_unet_mask_a1b2_best.pth \
    --data_dir segdata/DOTA/DOTA_PLANES/test \
    --save_dir inference_results \
    --threshold 0.3
```

L'inférence utilise une fenêtre glissante avec mélange gaussien pour traiter des images de n'importe quelle taille.

### Options

| Argument | Description |
|----------|-------------|
| `--ckpt` | Chemin vers le checkpoint |
| `--data_dir` | Dossier contenant `image/` et `mask/` |
| `--save_dir` | Dossier de sortie |
| `--threshold` | Seuil de détection (mode center) |
| `--tile_size` | Taille des tuiles (défaut: 512) |
| `--stride` | Pas de la fenêtre glissante (défaut: 384) |
| `--batch_size` | Batch size pour l'inférence |

### Sortie

Les visualisations côte-à-côte sont générées :
- **Gauche** : Ground truth (vert)
- **Droite** : Prédictions (orange)

## Modes de fonctionnement

### Mode Mask (segmentation)

```bash
--target_type mask --loss combo
```

Produit un masque binaire de segmentation. Métriques : Dice, IoU.

### Mode Center (détection de centres)

```bash
--target_type center --loss mse --sigma 8.0
```

Produit une heatmap avec des pics gaussiens aux centres des objets. Utile pour compter les objets.

## Structure du code

```
segdino/
├── model.py      # DINOv3 backbone + décodeurs (heavy_unet, medium, light)
├── dataset.py    # Dataset pré-tuilé avec support mask/center
├── loss.py       # Fonctions de perte (Dice, Combo, MSE, Focal)
├── train.py      # Script d'entraînement
├── inference.py  # Inférence avec fenêtre glissante
├── utils/
│   └── metrics.py    # Calcul Dice/IoU
└── pyproject.toml    # Dépendances
```

## Dépendances

Gérées via `uv` et `pyproject.toml` :

- PyTorch >= 2.3
- Transformers >= 4.40 (pour DINOv3)
- OpenCV
- NumPy, SciPy, tqdm

## Troubleshooting

### Out of Memory

Réduire `--batch_size` ou utiliser un backbone plus petit (`base`, `small-plus`).

### Token HuggingFace invalide

Vérifier que le fichier `.env` contient `HF_TOKEN=hf_xxxxx` et que vous avez demandé l'accès aux modèles DINOv3 sur HuggingFace.

### Masques non trouvés

Les masques doivent avoir exactement le même nom que les images correspondantes.
