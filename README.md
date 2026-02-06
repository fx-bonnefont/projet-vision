# SegDino - Center Detection

Detection de centres d'objets sur vues aeriennes avec DINOv3 et decodeur leger.

## Architecture

### Entrainement (features cachees)

```
Image RGB (512x512)
        |
        v
+-------------------+
|  DINOv3 Backbone  |  (gele, pre-entraine)
+--------+----------+
         | 4 feature maps (extraits une seule fois, caches sur disque)
         v
+-------------------+
|   Light Decoder   |  (~500K params, entrainable)
+--------+----------+
         |
         v
Heatmap de centres (512x512)
```

L'entrainement utilise des features pre-extraites pour eviter de charger le backbone a chaque epoque. Les features sont cachees automatiquement au premier lancement.

### Inference (modele complet)

A l'inference, le modele complet (backbone + decoder) est charge depuis le checkpoint.

## Installation

### Prerequis

- Python >= 3.10
- GPU CUDA (recommande) ou Apple Silicon (MPS)
- Token HuggingFace pour acceder aux modeles DINOv3

### Setup

```bash
git clone <repo-url>
cd segdino

# Installer les dependances
uv sync

# Configurer le token HuggingFace
echo "HF_TOKEN=hf_xxxxx" > .env
```

Le token HuggingFace est necessaire car les modeles DINOv3 sont en acces restreint.

## Structure des donnees

### Donnees d'entrainement (tiles 512x512)

Chemin hardcode : `segdata/DOTA/DOTA_PLANES_TILED/`

```
segdata/DOTA/DOTA_PLANES_TILED/
├── train/
│   ├── image/
│   │   ├── tile_0001.png
│   │   └── ...
│   └── mask/
│       ├── tile_0001.png   (meme nom que l'image)
│       └── ...
└── test/
    ├── image/
    └── mask/
```

- Images : RGB, 512x512, PNG
- Masques : Grayscale, 512x512, PNG (blanc = objet, noir = fond)

### Donnees d'inference (images completes)

Chemin hardcode : `segdata/DOTA/DOTA_PLANES/test/`

```
segdata/DOTA/DOTA_PLANES/test/
├── image/
│   ├── P0001.png   (n'importe quelle taille)
│   └── ...
└── mask/
    ├── P0001.png
    └── ...
```

## Entrainement

```bash
python train.py -m large-sat -e 20 -l 5e-4
```

### Options

| Flag | Valeurs | Description |
|------|---------|-------------|
| `-m` | `small`, `small-plus`, `base`, `large`, `huge`, `giant`, `large-sat`, `giant-sat` | Backbone DINOv3 (defaut: `small`) |
| `-e` | int (defaut: 20) | Nombre d'epoques |
| `-l` | float (defaut: 5e-4) | Learning rate |
| `-b` | int (defaut: 12) | Batch size |

### Backbones recommandes

| Backbone | Usage | VRAM |
|----------|-------|------|
| `large-sat` | **Production** - Pre-entraine sur images satellite | ~16 GB |
| `base` | Developpement rapide | ~8 GB |
| `small-plus` | Ressources limitees | ~4 GB |

### Sorties

Les checkpoints sont sauvegardes dans `runs/` avec un codename aleatoire :

```
runs/
├── 03_02-14_30_00_LARGE-SAT_swift-falcon.pth   # Modele (best F1)
└── 03_02-14_30_00_LARGE-SAT_swift-falcon.csv   # Logs (train_loss, val_loss, f1)
```

## Inference

```bash
python inference.py -c runs/03_02-14_30_00_LARGE-SAT_swift-falcon.pth
```

L'inference utilise une fenetre glissante avec melange gaussien pour traiter des images de n'importe quelle taille.

### Sorties

Les visualisations cote-a-cote sont generees dans `inference_results/` :
- **Gauche** : Ground truth (croix vertes)
- **Droite** : Predictions (croix oranges)

Un fichier `metrics.csv` contient les metriques (precision, recall, F1) a differents seuils.

## Structure du code

```
segdino/
├── model.py              # DINOv3 backbone + Light decoder + DecoderOnly
├── dataset.py            # Datasets (PreTiled + CachedFeatures)
├── loss.py               # CenterLoss (MSE + Focal)
├── train.py              # Entrainement (decoder seul sur features cachees)
├── inference.py          # Inference avec fenetre glissante
├── extract_features.py   # Extraction et cache des features backbone
├── codenames.py          # Generateur de noms aleatoires pour les runs
└── pyproject.toml        # Dependances
```

## Dependances

- PyTorch >= 2.5
- Transformers >= 4.40 (DINOv3)
- HuggingFace Hub >= 0.21
- Torchvision >= 0.20
- OpenCV (headless)
- NumPy >= 1.26 (< 2.0), SciPy, tqdm
- python-dotenv

## Troubleshooting

### Out of Memory

Utiliser un backbone plus petit (`base`, `small-plus`).

### Token HuggingFace invalide

Verifier `.env` contient `HF_TOKEN=hf_xxxxx` et demander l'acces aux modeles DINOv3 sur HuggingFace.

### Masques non trouves

Les masques doivent avoir exactement le meme nom que les images.
