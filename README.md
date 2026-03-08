# SegDino - Center Detection and Vit patch attacks

Detection de centres d'objets sur vues aeriennes avec DINOv3 et decodeur leger, assorti d'un système d'entrainement de patch d'attaque.

## Quick Start

```bash
# 1. Entrainer le decodeur sur les features DINOv3
python train.py -m large-sat -e 20 -l 5e-4 -b 12

# 2. Lancer l'inference sur des images completes (fenetre glissante)
python inference.py -c runs/<checkpoint>.pth

# 3. Entrainer un patch adversarial contre le modele
python train_patch.py -c runs/<checkpoint>.pth -e 50 --patch-size 16 --px 0 --py 0

# 4. Evaluer l'impact du patch sur le recall par image
python attack.py -c runs/<checkpoint>.pth --patch attack_results/<run_id>/patch.pt --patch-size 16
```

| Commande | Description |
|----------|-------------|
| `train.py` | Entraine le decodeur leger sur des features backbone cachees. Sauvegarde le meilleur modele (F1) dans `runs/`. |
| `inference.py` | Inference sur images completes de taille arbitraire avec fenetre glissante et melange gaussien. |
| `train_patch.py` | Optimise un petit patch RGB qui minimise les detections (attaque false negative). |
| `attack.py` | Evalue un patch entraine en comparant le recall clean vs patched par image. |
| `extract_features.py` | Pre-extrait les features backbone sur disque (fait automatiquement par `train.py`). |

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

## Utilisation

### Flags communs

| Flag | Description | Utilise par |
|------|-------------|-------------|
| `-m` | Backbone DINOv3 | `train.py`, `extract_features.py` |
| `-c` | Chemin du checkpoint | `inference.py`, `train_patch.py`, `attack.py` |
| `-e` | Nombre d'epoques | `train.py`, `train_patch.py` |
| `-l` | Learning rate | `train.py`, `train_patch.py` |
| `-b` | Batch size | `train.py`, `extract_features.py`, `train_patch.py`, `attack.py` |

### Entrainement du decodeur

```bash
python train.py -m large-sat -e 20 -l 5e-4 -b 12
```

| Flag | Valeurs | Description |
|------|---------|-------------|
| `-m` | `small`, `small-plus`, `base`, `large`, `huge`, `giant`, `large-sat`, `giant-sat` | Backbone DINOv3 (defaut: `small`) |
| `-e` | int (defaut: 20) | Nombre d'epoques |
| `-l` | float (defaut: 5e-4) | Learning rate |
| `-b` | int (defaut: 12) | Batch size |

Les checkpoints sont sauvegardes dans `runs/` avec un codename aleatoire :

```
runs/
├── 03_02-14_30_00_LARGE-SAT_swift-falcon.pth   # Modele (best F1)
└── 03_02-14_30_00_LARGE-SAT_swift-falcon.csv   # Logs (train_loss, val_loss, f1)
```

### Extraction manuelle des features

Les features sont cachees automatiquement au premier entrainement. Pour les pre-extraire manuellement :

```bash
python extract_features.py -m large-sat -b 16
```

| Flag | Description |
|------|-------------|
| `-m` | Backbone DINOv3 (requis) |
| `-b` | Batch size (defaut: 16) |
| `--data-dir` | Chemin des tiles (defaut: `segdata/DOTA/DOTA_PLANES_TILED`) |
| `--fp32` | Sauvegarder en float32 au lieu de float16 |

### Inference

```bash
python inference.py -c runs/03_02-14_30_00_LARGE-SAT_swift-falcon.pth
```

L'inference utilise une fenetre glissante avec melange gaussien pour traiter des images de n'importe quelle taille. Les visualisations cote-a-cote sont generees dans `inference_results/` :
- **Gauche** : Ground truth (croix vertes)
- **Droite** : Predictions (croix oranges)

Un fichier `metrics.csv` contient les metriques (precision, recall, F1) a differents seuils.

### Attaque adversariale (patch)

#### Entrainement du patch

```bash
python train_patch.py -c runs/03_02-14_30_00_LARGE-SAT_swift-falcon.pth -e 50 -l 0.1 --patch-size 16 --px 0 --py 0
```

| Flag | Description |
|------|-------------|
| `-c` | Checkpoint du modele cible (requis) |
| `-e` | Nombre d'epoques (defaut: 50) |
| `-l` | Learning rate (defaut: 0.1) |
| `-b` | Batch size (defaut: 4) |
| `--patch-size` | Taille du patch en pixels (defaut: 16) |
| `--px`, `--py` | Position du patch sur la tile (defaut: 0, 0) |
| `--threshold` | Seuil de detection (defaut: 0.3) |
| `--temperature` | Temperature du sigmoid (defaut: 10.0) |
| `--limit` | Nombre max de tiles a utiliser |

Les resultats sont sauvegardes dans `attack_results/<run_id>/` : `patch.pt`, `patch.png`, `metrics.csv`.

#### Evaluation du patch

```bash
python attack.py -c runs/03_02-14_30_00_LARGE-SAT_swift-falcon.pth --patch attack_results/<run_id>/patch.pt --patch-size 16 --px 0 --py 0
```

| Flag | Description |
|------|-------------|
| `-c` | Checkpoint du modele cible (requis) |
| `--patch` | Chemin du patch entraine `.pt` (requis) |
| `-b` | Batch size (defaut: 4) |
| `--patch-size` | Taille du patch (defaut: 16) |
| `--px`, `--py` | Position du patch (defaut: 0, 0) |
| `--threshold` | Seuil de detection (defaut: 0.3) |

Affiche le recall clean vs patched par image parente.

## Backbones recommandes

| Backbone | Usage | VRAM |
|----------|-------|------|
| `large-sat` | **Production** - Pre-entraine sur images satellite | ~16 GB |
| `base` | Developpement rapide | ~8 GB |
| `small-plus` | Ressources limitees | ~4 GB |

## Structure du code

```
segdino/
├── model.py              # DINOv3 backbone + Light decoder + DecoderOnly
├── dataset.py            # Datasets (PreTiled + CachedFeatures)
├── loss.py               # CenterLoss (MSE + Focal)
├── train.py              # Entrainement (decoder seul sur features cachees)
├── inference.py          # Inference avec fenetre glissante
├── extract_features.py   # Extraction et cache des features backbone
├── train_patch.py        # Entrainement du patch adversarial
├── attack.py             # Evaluation du patch adversarial
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

### Références

@ARTICLE{9560031, 
    author={Ding, Jian and Xue, Nan and Xia, Gui-Song and Bai, Xiang and Yang, Wen and Yang, Michael and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei}, 
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
    title={Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges}, 
    year={2021}, 
    volume={}, 
    number={}, 
    pages={1-1}, 
    doi={10.1109/TPAMI.2021.3117983}
    }

@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut, Patrick and Bojanowski, Piotr},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.10104},
}