import os
import cv2
import numpy as np

MASK_DIR = "data/DOTA/DOTA_PLANES_TILED/train/mask"

polygon_areas = []          # (filename, area)
single_plane_files = []     # (filename, area)
zero_area_files_count = 0   # compteur fichiers avec polygone(s) de taille 0
zero_area_files = []
zero_plane_files = []

for filename in os.listdir(MASK_DIR):
    if not filename.endswith(".png"):
        continue

    path = os.path.join(MASK_DIR, filename)

    # Lecture du masque
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Seuillage
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Trouver tous les contours (tous les polygones)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Si aucun polygone → fichier vide
    if len(contours) == 0:
        zero_plane_files.append((filename))
        if "_obj_" in filename:
            zero_area_files_count += 1
        continue

    # Calcul des aires individuelles
    areas = [cv2.contourArea(cnt) for cnt in contours]

    # Comptage des polygones de taille 0
    if any(a == 0 for a in areas) and "_obj_" in filename:
        zero_area_files_count += 1
        zero_area_files.append((filename, areas[0]))

    # Stockage des polygones individuels
    for a in areas:
        polygon_areas.append((filename, a))

    # Stockage des fichiers avec un seul avion
    if len(areas) == 1:
        single_plane_files.append((filename, areas[0]))

# --- Analyse des polygones individuels ---

# Tri par aire individuelle
polygon_areas_sorted = sorted(polygon_areas, key=lambda x: x[1])

smallest = polygon_areas_sorted[0]
largest = polygon_areas_sorted[-1]

# Moyenne globale
mean_area = np.mean([a[1] for a in polygon_areas])

# Polygone le plus proche de la moyenne
closest = min(polygon_areas, key=lambda x: abs(x[1] - mean_area))

# --- Fichier avec un seul avion le plus proche de la moyenne ---

if single_plane_files:
    closest_single = min(single_plane_files, key=lambda x: abs(x[1] - mean_area))
else:
    closest_single = None

# --- Résultats ---

print("===== RÉSULTATS =====")
print(f"Plus petit polygone : {smallest[0]}  -> {smallest[1]:.2f} px")
print(f"Plus grand polygone : {largest[0]}   -> {largest[1]:.2f} px")
print(f"Aire moyenne globale : {mean_area:.2f} px")
print(f"Polygone le plus proche de la moyenne : {closest[0]} -> {closest[1]:.2f} px")

if closest_single:
    print(f"Fichier avec un seul avion le plus proche de la moyenne : {closest_single[0]} -> {closest_single[1]:.2f} px")
else:
    print("Aucun fichier avec un seul avion trouvé.")

print(f"Nombre de fichiers avec polygones de taille 0 px : {zero_area_files_count}")
#print(f"Fichiers avec zéro avions : {zero_plane_files}")
print(zero_area_files)