# Exemplar-Based inpainting project

This project implements the exemplar-based inpainting implementation in `inpainting.pdf`.

The main algorithm is in `src/main.py`. You can run directly the file and modify which image to use in the `if __name__="__main__"` section.

The `data/` directory countains multiple images and their masks in order to test our algorithm. You can also choose to create your own rectangular mask by having `create_mask=1` in the `main.py` algorithm.

If you save the inpainted image, you'll find it in `output/`.

#

Ce projet implémente la méthode d'exemplar-based inpainting qui se trouve dans `inpainting.pdf`.

L'algorithme principal se trouve dans `main.py`. Vous pouvez exécuter directement le fichier et vous pouvez aussi modifier quelle image utiliser dans la section `if __name__="__main__"`.

Le dossier `data/` contient plusieurs images et leurs masques pour tester notre algorithme. Vous pouvez aussi choisir de créer vos propres masques rectangulaires en mettant `create_mask=1` dans l'algorithme de `main.py`.

Si vous sauvegardez l'image remplie, vous pourrez la trouver dans `output/`.
