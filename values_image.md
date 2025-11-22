# Les valeurs à mettre pour correctement remplir une image

Pour notre algorithme, il y a quelques paramètres à ajuster en fonction de l'image pour avoir un bon résultat :

- `patch_size`
- `search_radius`
- `sigma_lissage` (dans `utils`)

### Dog Example

- Patch_size = 4
- search radius = img \* 0.40
- sigma_lissage = 1.0

### Triangle Simple

- patch_size = 7
- search_prop = 0.35
- sigma_lissage = 1.25
