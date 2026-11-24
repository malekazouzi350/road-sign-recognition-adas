# Détection et reconnaissance des panneaux de signalisation (ADAS)

Projet universitaire de vision par ordinateur : trouver les panneaux de signalisation dans une
image de route, puis reconnaître leur type parmi 43 classes, avec une application de démonstration.

> **Prototype académique.** Ce système n'est pas certifié et ne doit pas servir à contrôler un
> vrai véhicule.

## Principe : deux étapes

```
image de route ──► détecteur YOLO26n ──► boîtes + famille ──► CNN de l'équipe ──► nom du panneau
                   (4 familles :           (interdiction,       (43 classes,        ou « famille (?) »
                    entraîné sur GTSDB)     danger, obligation,   entraîné sur        si la confiance
                                            autre)                GTSRB)              du CNN < 0,6
```

1. **Détecteur** : YOLO26n (Ultralytics), pré-entraîné sur COCO puis ré-entraîné par l'équipe sur
   GTSDB pour trouver les panneaux et leur famille.
2. **Reconnaissance** : un CNN écrit et entraîné par l'équipe (1,48 M paramètres, entrée 48×48 px)
   donne la classe exacte parmi 43.
3. **Prudence** : si le CNN est sûr à moins de 60 %, l'application affiche seulement la famille
   suivie de « (?) » au lieu d'inventer un nom.

## Résultats mesurés

Tous les chiffres sont lus dans `results/` et produits par les scripts de `scripts/`. Le test se
fait sur les **300 images de test officielles de GTSDB (361 panneaux)**, jamais vues pendant
l'entraînement.

| Élément | Mesure | Résultat | Fichier |
|---|---|---|---|
| YOLO générique (COCO, sans ré-entraînement) | panneaux trouvés | 10,4 % | `results/baseline/` |
| CNN v1 sur GTSRB (5 850 images de test) | accuracy, 43 classes | 99,69 % | `results/classifier/` |
| **CNN v2** (entraîné avec flou de bougé) sur GTSRB | accuracy, 43 classes | **99,78 %** | `results/classifier_blur/` |
| CNN v2 sur les panneaux découpés de GTSDB | accuracy (boîtes exactes / décalées de ±15 %) | 99,67 % / 98,68 % | `results/classifier_on_gtsdb/` |
| Détecteur v1 (1024 px) | rappel / mAP50 | 95,3 % / 97,7 % | `results/detector/v1_quick/` |
| Détecteur v2 (1280 px) | rappel / mAP50 | 97,4 % / 98,9 % | `results/detector/v2_1280/` |
| Système complet avec détecteur v1 | panneaux trouvés **et** bien nommés | 98,3 % (355/361), 21 fausses alarmes | historique Git de `results/pipeline/` |
| **Système complet** (détecteur v2 + CNN v2) | panneaux trouvés **et** bien nommés | **98,9 %** (357/361) | `results/pipeline/` |
| Système complet (v2) | mauvais noms | **0** | `results/pipeline/` |
| Système complet (v2) | fausses alarmes | 25 (93,5 % des boîtes sont des panneaux annotés) | `results/pipeline/` |

À savoir pour lire ces chiffres :

- Le rappel du détecteur (protocole Ultralytics, la famille doit être juste) et celui du système
  complet (boîte trouvée, puis classe exacte vérifiée par le CNN) ne se mesurent pas de la même
  façon. Les deux sont valables, mais il faut dire lequel on cite.
- Le détecteur v2 (1280 px) est utilisé par défaut : il trouve 3 panneaux de plus que v1 (358 au lieu
  de 355), surtout des petits (moins de 32 px : 96,9 % au lieu de 93,8 %), au prix de 4 fausses
  alarmes de plus (25 au lieu de 21).
- Une partie des « fausses alarmes » sont de vrais panneaux qui ne font pas partie des 43 classes
  annotées (arrêt interdit, arrêt de bus…). Le chiffre de 93,5 % est donc prudent.
- Le CNN v2 a remplacé le CNN v1 parce qu'il remplissait les 3 conditions fixées avant l'essai :
  au moins aussi bon sur GTSRB, même résultat de bout en bout sur les images nettes (98,9 %,
  0 mauvais nom), et moins de mauvais noms sous le flou (voir ci-dessous). Sur les boîtes décalées
  de ±15 %, il est un peu moins bon que v1 (98,68 % au lieu de 99,34 %).

### Robustesse (conditions simulées)

Les 300 images de test sont dégradées artificiellement (`scripts/evaluate_robustness.py`) : ce sont
des **simulations**, pas de vraies photos de nuit ou à contre-jour. Pourcentage de panneaux trouvés
**et** bien nommés, système complet (détecteur v2 + CNN v2) :

| Condition simulée | Léger | Moyen | Fort | Mauvais noms (léger / moyen / fort) |
|---|---|---|---|---|
| Images d'origine | 98,9 % | | | 0 |
| Voile clair (brouillard, contre-jour) | 97,0 % | 94,2 % | 80,3 % | 0 / 0 / 0 |
| Bruit du capteur | 94,5 % | 75,9 % | 47,4 % | 0 / 0 / 0 |
| Flou de bougé | 94,5 % | 82,5 % | 47,1 % | 0 / 2 / 8 |
| Nuit (image sombre + bruit) | 93,9 % | 70,1 % | 31,0 % | 0 / 0 / 1 |

Avec le CNN v1, le flou de bougé était la seule condition qui produisait de vrais mauvais noms
(12 au niveau moyen, 18 au niveau fort). Entraîner le CNN avec du flou de bougé simulé les a réduits
à 2 et 8, sans rien perdre sur les images nettes. En contrepartie, avec le voile clair fort, le CNN v2
affiche plus souvent « (?) » (80,3 % au lieu de 85,9 %, toujours sans mauvais nom). Détails, graphique
et exemples : `results/robustness/` et l'onglet « Robustesse » de la page Évaluation.

## Installation (Windows, PowerShell)

Il faut **Python 3.12**. Ne mettez pas pip à jour dans l'environnement du projet.

```powershell
git clone https://github.com/malekazouzi350/road-sign-recognition-adas.git
cd road-sign-recognition-adas
python -m venv .venv
.venv\Scripts\activate
```

Installez **PyTorch en premier**, selon votre machine :

```powershell
# Carte NVIDIA (pilote >= 580)
pip install "torch==2.15.*" "torchvision==0.29.*" --index-url https://download.pytorch.org/whl/cu132
# Sans carte NVIDIA
pip install "torch==2.15.*" "torchvision==0.29.*" --index-url https://download.pytorch.org/whl/cpu
```

Puis le reste du projet, et une vérification :

```powershell
pip install -r requirements.txt
pip install -e .
python scripts\check_env.py
pytest -q
```

### Données et poids des modèles

Ils ne sont **pas dans Git** (trop lourds, licences propres).

- **Données** : suivre [`data/README.md`](data/README.md) pour télécharger GTSDB et GTSRB dans
  `data/raw/`.
- **Poids** : récupérer sur le Google Drive de l'équipe et placer dans `models/weights/` :
  - `detector_v2_1280.pt` (détecteur utilisé par l'application) ;
  - `classifier_blur.pt` (CNN v2 utilisé par l'application).

## Lancer l'application

```powershell
streamlit run app/app.py
```

Le navigateur s'ouvre sur `http://localhost:8501`.

| Page | Ce qu'elle fait |
|---|---|
| Tableau de bord | résultats mesurés, modèles et données présents sur l'ordinateur, analyses de la session |
| Détection › Sur image | importer une photo ou choisir une image de test GTSDB ; image annotée, panneaux découpés, tableau, téléchargements |
| Détection › Sur vidéo | vidéo annotée, une image analysée sur N, suivi de chaque panneau (nom décidé par vote, numéro affiché), liste des panneaux avec une colonne de fiabilité |
| Détection › Webcam | photo avec la caméra : mode « scène de route » (détecteur + CNN) ou « panneau en gros plan » (CNN seul) |
| Évaluation | toutes les mesures et courbes de `results/` |
| Paramètres | seuils du détecteur et du CNN, taille d'image, CPU ou GPU |

Sans GPU, comptez environ 0,5 à 1 s par image (en 1280 px ; passez à 1024 px dans Paramètres pour aller plus vite). Les pages affichent un message clair si un fichier de
poids manque.

## Reproduire les résultats

```powershell
python scripts\baseline_pretrained.py                        # YOLO COCO sans ré-entraînement
python scripts\train_classifier.py                           # CNN v1 sur GTSRB
python scripts\train_classifier.py --motion-blur --weights-out models\weights\classifier_blur.pt --out results\classifier_blur   # CNN v2
python scripts\evaluate_classifier_gtsdb.py                  # CNN v2 sur les panneaux de GTSDB
python scripts\prepare_gtsdb.py                              # GTSDB -> format YOLO (510 / 90 / 300)
python scripts\train_detector.py --name v1_quick --epochs 25 # détecteur v1 (GTX 1650 : sans --amp)
python scripts\train_detector.py --name v2_1280 --imgsz 1280 --batch 4 --amp --epochs 100
python scripts\evaluate_pipeline.py                          # système complet
python scripts\evaluate_robustness.py                        # robustesse (conditions simulées)
```

Les essais rapides vont dans `results/tmp/`, qui n'est pas suivi par Git.

## Organisation du dépôt

```
app/                 application Streamlit (pages, composants, style)
src/roadsigns/       code du projet : datasets, CNN, pipeline, vidéo, évaluation
scripts/             entraînement, évaluation, préparation des données
tests/               tests automatiques (pytest)
notebooks/           exploration de GTSDB
results/             mesures et figures produites par les scripts
data/                instructions de téléchargement (les données restent en local)
models/              poids (en local, hors Git)
```

## Limites connues

- Les modèles ont appris sur des **panneaux allemands**. Les panneaux d'autres pays (fond jaune,
  formes différentes) sont souvent affichés en « (?) ». Les panneaux tunisiens ne sont pas encore
  évalués.
- Le détecteur a appris des panneaux **petits** dans des scènes de route (médiane 39 px). Il ne
  trouve pas un panneau qui remplit toute l'image : c'est le rôle du mode « gros plan » de la page
  webcam.
- En vidéo, le suivi d'un panneau d'une image à l'autre se fait seulement par la position des boîtes.
  Un panneau caché un moment peut recevoir deux numéros. Faute de vidéos annotées, ce suivi est
  vérifié par des tests et à l'œil, pas mesuré.
- Les conditions difficiles (nuit, flou, brouillard, bruit) sont seulement **simulées** sur les images
  de test : il n'y a pas de vraies photos de nuit annotées. La nuit forte et le bruit fort font surtout
  rater des panneaux (31 % et 47 % de réussite au niveau fort).

## Équipe

- **Malek Azzouzi** : modèles (détecteur, CNN, pipeline)
- **Nourhene Benkhessib** : données et application
- **Tasbih Soussi** : entraînement GPU à 1280 px, table des panneaux, jeu de test tunisien

## Licences

- Ultralytics YOLO : AGPL-3.0.
- GTSDB et GTSRB : voir les conditions et la citation dans [`data/README.md`](data/README.md).
