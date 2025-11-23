Notes - 1 -  22/09 : image processing gradient: sobel
ne pas essayer d'etre efficace dès le début
peu de boucles sur python, utiliser vecteurs (ou patch oula m3rte)
on peut commencer par le coeur de l'algorithme: sans priorité, méthode gloutonne
après, on aura un problème de calcul de gradient: il y aura des valeurs inconnues et c à nous de voir comment faire
c mieux de tester des images géométriques
commencer sur de petites images pour debugger
il faut décider le mask

-

Tester sur des figures géométriques
ajouter le calcul de priorité
ajouter une interface interactive pour créer le masque
vérifier fonction qui traite le masque (different masque dans le plot que l'output)
save image ne marche pas (à la fin de l'algorithme)

-
savoir pourquoi pour le terme de confiance seulement l'ordre de remplissage n'était pas correct
essayer sur des *images géométriques simples* (important)
essayer sans data term

-
ajouter progress bar (tqdm)
le temps d'execution dépend bcp de la résolution (surtt dans alternative avec RGB)
