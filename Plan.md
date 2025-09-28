0) Organisation du binôme & des RDV
	•	Rôles (réversibles)
	•	Binôme A – Algorithme & core code : front de remplissage, priorité, recherche d’exemplaires, optimisation.
	•	Binôme B – Expériences & doc : jeux de données/masques, métriques (PSNR/SSIM), figures, rapport/vidéo.
	•	2 créneaux hebdo courts (30–45 min) : stand‑up (lundi) + sync code/expés (jeudi).
	•	Encadrants
	•	3 points fixes : S1 (validation plan), S4 (revue avant rapport intermédiaire), S8 (relecture finale).
	•	Gestion
	•	Repo Git (branches: algo/, expes/, docs/).
	•	Dossier data (images + masques).
	•	Feuille de route simple (Kanban “À faire / En cours / Fait”).

⸻

1) Références techniques clefs (à implémenter)

Vous implémentez l’algorithme exemplar‑based inpainting :
	•	Priorité des patchs remplissage “best‑first” : P(p)=C(p)\,D(p) où
C(p) = confiance moyenne des pixels déjà connus dans le patch,
D(p)=\frac{\lvert \nabla I_p^\perp \cdot n_p \rvert}{\alpha} met en avant la poursuite des isophotes (structure). Voir l’équation (1) et le schéma de notation (Fig. 3, p. 3).  ￼
	•	Sélection du patch à remplir : on choisit le patch sur le front \delta\Omega ayant la priorité max (pseudo‑code, Table 1, p. 4).  ￼
	•	Copie d’exemplaire (texture + structure) : on cherche dans \Phi le patch entierement connu \Psi_{\hat q} minimisant la SSD sur les pixels déjà connus du patch cible \Psi_{\hat p} (équation (2), p. 4), puis on copie les valeurs manquantes ; couleur en Lab recommandée.  ￼
	•	Mise à jour : les confiances des pixels nouvellement remplis prennent C(\hat p) (Table 1, p. 4). Taille de patch : par défaut 9×9, à ajuster “un peu plus grand que le plus gros texel/épaisseur de structure” (p. 2).  ￼

Astuce pour la soutenance : appuyez‑vous visuellement sur Fig. 4 p. 4 (progression du front), Fig. 7 p. 5 (avantage vs “onion peel”) et Fig. 10 p. 7 (gain vs inpainting diffusif).  ￼

⸻

2) Planning daté (9 semaines ≈ 30 h)

Jalons école :
Lun. 15/09 mise en ligne – Mer. 17/09 choix du sujet – Mer. 15/10 rapport intermédiaire (1 page) – Semaine du 17/11 soutenances.

Semaine 1 (15–21/09) — Kick‑off & bases (4–5 h)
	•	Lire/annoter l’article (p. 2–4 pour l’algo, Table 1 p. 4). Préparer 5 slides de design.  ￼
	•	Créer repo, environnement (Python ≥ 3.10, NumPy, OpenCV, scikit‑image, Matplotlib).
	•	Collecte d’images (textures, scènes naturelles) + génération de masques (brosses libres + masques synthétiques).
	•	RDV encadrant #1 : valider le plan, les critères d’évaluation, et le jeu d’images.

Semaine 2 (22–28/09) — Masque & front, gradients (3–4 h)
	•	Lecture/écriture masques, extraction du front \delta\Omega (contour du masque).
	•	Gradients Sobel, calcul isophotes \nabla I^\perp, normales n_p au front.
	•	Carte de confiance C initiale (1 hors masque / 0 dans le masque).

Semaine 3 (29/09–05/10) — Priorité & sélection (3–4 h)
	•	Implémenter P(p)=C(p)D(p) par patchs centrés sur les pixels du front.
	•	Sélecteur du patch à priorité max (gestion des bordures, tailles impaires).

Semaine 4 (06–12/10) — Recherche d’exemplaires & copie (4–5 h)
	•	Recherche brute de \Psi_{\hat q} (SSD en Lab) dans \Phi (I \setminus \Omega).
	•	Copie des pixels inconnus de \Psi_{\hat p} depuis \Psi_{\hat q}.
	•	Maj confiances + maj masque/front – boucle jusqu’à fermeture du trou.
	•	Premières images de résultats (figures “avant/pendant/après”).

Semaine 5 (13–19/10) — Rapport intermédiaire & démo (3 h)
	•	Rapport 1 page (15/10) :
	1.	brève description du sujet,
	2.	ce qui est fait (pipeline, premières images),
	3.	planning restant (S6–S9),
	4.	points de risque.
	•	RDV encadrant #2 : retours + priorités pour la suite.

Semaine 6 (20–26/10) — Qualité visuelle & robustesse (4 h)
	•	Étude taille patch (7–11–15) et fenêtre de recherche (bande autour de \Omega vs plein \Phi).
	•	Fondu léger aux frontières (moyenne pondérée / lissage local) pour réduire les seams.
	•	Cas difficiles : structures épaisses, textures non stationnaires (tester masques de formes variées).

Semaine 7 (27/10–02/11) — Vitesse & variantes (3–4 h)
	•	Accélération SSD (intégrales de carrés, pré‑filtrage, downsampling pyramidal).
	•	Option : carte de salience pour restreindre les candidats (bordures/coins).
	•	Sauvegarde des artefacts typiques pour la partie “Limites”.

Semaine 8 (03–09/11) — Évaluation & rapport final (4 h)
	•	Métriques (si GT accessible : PSNR/SSIM entre image originale et “image masquée puis inpaintée”).
	•	Tableaux comparatifs taille de patch / fenêtre de recherche / temps.
	•	Rédaction du rapport final (structure ci‑dessous), figures propres.

Semaine 9 (10–16/11) — Vidéo & dry‑run (3–4 h)
	•	Vidéo 10 min (scénario ci‑dessous), enregistrement + sous‑titres.
	•	RDV encadrant #3 : répétition de soutenance (Q/R 15 min).
	•	Freeze du code + artefacts reproductibles.

⸻

3) Détails d’implémentation (check‑list)
	1.	Entrées : image RGB → Lab pour les SSD (recommandé dans l’article, p. 4).  ￼
	2.	Masque binaire \Omega, front \delta\Omega par gradient morphologique.
	3.	Priorité P(p) :
	•	C(p) = moyenne des confiances déjà remplies dans le patch \Psi_p (0…1).
	•	D(p) : calculer \nabla I (Sobel), former \nabla I^\perp, normaliser par \alpha (255 si 8 bits), projeter sur n_p.
	4.	Sélection du patch \Psi_{\hat p} de priorité max.
	5.	Recherche exemplar \Psi_{\hat q} dans \Phi : SSD sur les pixels connus seulement de \Psi_{\hat p} (équation (2)).  ￼
	6.	Copie des pixels inconnus, maj C sur \Psi_{\hat p}, maj masque/front.
	7.	Boucle jusqu’à \Omega=\varnothing.
	8.	Options :
	•	bande \Phi dilatée autour de \Omega (réduit les faux‑matchs lointains),
	•	patch multitaille (grand → structure, petit → texture),
	•	post‑blend local/Poisson pour lisser les coutures.

⸻

4) Expérimentation & évaluation
	•	Jeux d’images : scènes naturelles, textures (eau, herbe, pierre), objets (façades lignes droites).
	•	Masques : trous convexes/concaves, bandes fines (rayures), grands objets.
	•	Mesures :
	•	Quantitatives (si on a la GT en masquant artificiellement) : PSNR/SSIM.
	•	Qualitatives : grille d’images Originale / Masquée / Étapes / Résultat, zooms sur les structures prolongées (cf. Fig. 7 & 13).  ￼
	•	Ablations : taille patch, fenêtre \Phi, stratégie de recherche, fondu.
	•	Comparatifs (si souhaité) : un baseline simple “onion peel” (remplissage concentrique) pour montrer l’intérêt de D(p) (cf. Fig. 7, p. 5).  ￼

⸻

5) Rapport intermédiaire (1 page – 15/10)
	•	Sujet : inpainting par patchs guidé par isophotes (Criminisi).
	•	Déjà réalisé : pipeline, priorité, premiers résultats (1 figure).
	•	Planning restant : S6–S9 avec livrables.
	•	Risques : temps de calcul, structures courbes, textures non stationnaires → contremesures (bande \Phi, patch adaptatif, pyramide).

⸻

6) Rapport final (structure conseillée, ~6–10 pages)
	1.	Introduction / cas d’usage (object removal, restauration).
	2.	Méthode (notation, P(p), SSD, boucle – renvoyer Table 1 p. 4 et Fig. 3 p. 3).  ￼
	3.	Implémentation (choix pratiques, tailles, accélérations).
	4.	Expériences (datasets, protocoles, métriques).
	5.	Résultats (tableaux, figures lisibles, temps d’exécution).
	6.	Discussion (forces/limites : cf. exemples Fig. 10 p. 7, Fig. 12–14 p. 8).  ￼
	7.	Conclusion & perspectives (vidéo, séquences vidéo si bonus).

⸻

7) Vidéo (10 min) & Soutenance (15 min Q/R)

Storyboard (10 min)
	1.	Problème & démos “avant/après” (30 s).
	2.	Idée clé : priorité P=C\cdot D, poursuite de structure (1 min 30).
	3.	Pipeline visuel (masque → front → patch prioritaire → exemplaire → copie) (2 min).
	4.	Paramètres & pièges (patch size, fenêtre, temps) (2 min).
	5.	Résultats & ablations (2 min).
	6.	Limites & perspectives (1 min).
	7.	Conclusion (30 s).
Q/R à préparer : normalisation de D(p), choix Lab, pourquoi SSD, compromis taille patch, complexité temporelle, comment éviter les “seams”.

⸻

8) Check‑lists rapides

Technique
	•	Front \delta\Omega stable (pas d’îlots).
	•	Priorité recalculée à chaque itération.
	•	SSD sur pixels connus uniquement.
	•	Confiances mises à jour et gelées après copie.
	•	Images “étapes” sauvegardées.

Expériences
	•	10–15 paires (image, masque) variées.
	•	2 tailles de patch min., 2 fenêtres \Phi, 1 fondu.
	•	Tableau temps/qualité.
	•	Figures lisibles (mêmes crops).

⸻

Petit mémo “page‑repère” de l’article
	•	p. 2 : taille de patch conseillée (≈ plus grand texel/structure).  ￼
	•	p. 3 : équation (1) P(p)=C(p)D(p), Fig. 3 notation.  ￼
	•	p. 4 : pseudo‑code Table 1, équation (2) (SSD).  ￼
	•	p. 5–8 : comparaisons “onion peel”/inpainting diffusif & gros objets (Figs. 7–14).  ￼
