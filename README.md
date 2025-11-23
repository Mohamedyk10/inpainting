# Exemplar-Based inpainting project

This project implements the exemplar-based inpainting implementation in `inpainting.pdf`.

The main algorithm is in `src/`. You can run directly the file and modify which image to use in the `if __name__="__main__"` section.

The `data/` directory countains multiple images and their masks in order to test our algorithm. You can also choose to create your own rectangular mask by having `create_mask=1` in the `main.py` algorithm.

If you save the inpainted image, you'll find it in `output/`.

There are 3 Versions of the algorithm :
- V1 : inpainting without dataterm.
- V2 : inpainting with dataterm.
- V3 : inpainting with dataterm optimized.
  
For the V3, if you want to correctly inpaint an image, you'll need to change the values of the following variables :

- `patch_size`
- `search_prop` (proportion of the image used in the updating of gradients around contour pixels 0<`search_prop`<1)
- `sigma_lissage` the power of the used smoothering filter in the algorithm.

You can find in `values_image.md` a list of optimal values for some images in the `data/` folder.

#

Ce projet implémente la méthode d'exemplar-based inpainting qui se trouve dans `inpainting.pdf`.

L'algorithme principal se trouve dans `src/`. Vous pouvez exécuter directement le fichier et vous pouvez aussi modifier quelle image utiliser dans la section `if __name__="__main__"`.

Le dossier `data/` contient plusieurs images et leurs masques pour tester notre algorithme. Vous pouvez aussi choisir de créer vos propres masques rectangulaires en mettant `create_mask=1` dans l'algorithme de `main.py`.

Si vous sauvegardez l'image remplie, vous pourrez la trouver dans `output/`.

Il y a 3 versions de l'algorithme :
- V1 : inpainting sans terme de données.
- V2 : inpainting avec terme de données.
- V3 : inpainting avec terme de données optimisé.

Pour la V3, si vous voulez remplir correctement une image, il vous faudra changer les valeurs des variables suivantes :

- `patch_size`
- `search_prop` (proportion de l'image utilisée dans la mise à jour des gradients autour des pixels du contour 0<`search_prop`<1)
- `sigma_lissage` la puissance du filtre lissant utilisé dans l'algorithme.

Vous pouvez trouver dans `values_image.md` une liste de valeurs optimales pour certaines images du dossier `data/`.
