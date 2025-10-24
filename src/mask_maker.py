import numpy as np
import matplotlib.pylab as plt
import os

# --- PARAMÈTRES DE VOTRE IMAGE ---
HAUTEUR = 280  # Nombre de lignes
LARGEUR = 800  # Nombre de colonnes
NOM_FICHIER_MASQUE = "entete-textures.mask.webp"

# --- DÉFINITION DE LA ZONE À MASQUER (EN PIXELS) ---
# Masque la zone centrale : de la ligne 80 à 200 et de la colonne 300 à 500.
Y1, Y2 = 33, 60  # Lignes (Hauteur)
X1, X2 = 488, 531  # Colonnes (Largeur)

# 1. Création du masque binaire (tout à zéro initialement)
masque = np.zeros((HAUTEUR, LARGEUR), dtype=np.uint8)

# 2. Définition de la zone à masquer (mise à 1)
# Note : Les coordonnées sont (ligne, colonne)
masque[Y1:Y2, X1:X2] = 1

# --- Enregistrement du masque ---
# Assurez-vous que le chemin vers le dossier 'data' est correct.
# Si ce script est exécuté depuis le dossier 'src' (comme main.py),
# le dossier 'data' est son voisin.
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))
mask_path = os.path.join(data_dir, NOM_FICHIER_MASQUE)

# Enregistrement du tableau numpy comme image (format .webp ou .png)
plt.imsave(mask_path, masque, cmap='gray')

print(f"Masque binaire ({HAUTEUR}x{LARGEUR}) créé et enregistré à : {mask_path}")
print(f"La zone masquée est : Lignes [{Y1}:{Y2}], Colonnes [{X1}:{X2}]")