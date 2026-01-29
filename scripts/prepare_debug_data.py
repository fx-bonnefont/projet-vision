import os

import cv2
import numpy as np

# Configuration des dossiers
IMAGE_DIR = './images'  # Dossier contenant vos .png
LABEL_DIR = './labels/DOTA-v1.5_train'  # Dossier contenant vos .txt
OUTPUT_DIR = './outputs/debug' # Dossier où seront sauvegardées les images dessinées

# Création du dossier de sortie s'il n'existe pas
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def draw_bboxes():
    # Liste toutes les images dans le dossier
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.png') or f.endswith('.jpg')]
    
    for img_name in image_files:
        img_id = os.path.splitext(img_name)[0]
        label_path = os.path.join(LABEL_DIR, f"{img_id}.txt")
        img_path = os.path.join(IMAGE_DIR, img_name)
        output_path = os.path.join(OUTPUT_DIR, f"visu_{img_name}")

        # Vérifier si l'image existe déjà
        if os.path.exists(output_path):
            print(f"Image déjà existante : {output_path}, ignorée.")
            continue

        if not os.path.exists(label_path):
            print(f"Label manquant pour {img_name}, ignoré.")
            continue

        # Lecture de l'image
        image = cv2.imread(img_path)
        if image is None:
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # Ignorer les lignes d'en-tête (source, gsd)
                if 'imagesource' in line or 'gsd' in line or not line.strip():
                    continue
                
                parts = line.strip().split()
                # Format DOTA : x1 y1 x2 y2 x3 y3 x4 y4 class difficulty
                if len(parts) >= 8:
                    coords = np.array([
                        [float(parts[0]), float(parts[1])],
                        [float(parts[2]), float(parts[3])],
                        [float(parts[4]), float(parts[5])],
                        [float(parts[6]), float(parts[7])]
                    ], np.int32)
                    
                    label = parts[8]
                    
                    # Choix de la couleur selon la classe
                    color = (0, 255, 0) # Vert par défaut
                    if 'plane' in label: color = (255, 0, 0) # Bleu pour les avions
                    if 'vehicle' in label: color = (0, 0, 255) # Rouge pour les véhicules
                    
                    # Dessiner le polygone (OBB)
                    cv2.polylines(image, [coords], isClosed=True, color=color, thickness=3)

        # Sauvegarde du résultat
        cv2.imwrite(output_path, image)
        print(f"Image générée : {output_path}")

if __name__ == "__main__":
    draw_bboxes()