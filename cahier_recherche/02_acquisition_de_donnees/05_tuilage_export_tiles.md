# 05 — Génération des vignettes & structure de `tiles/` (export du pilote)

> ⚠️ **Ce chapitre décrit l'export unique réalisé pour l'expérience pilote**, qui a produit les 582 tuiles de `tiles/`. Les tuiles s'y chevauchent à 50 % et le lot n'est pas découpé en volets : réparti après coup, il a produit la fuite spatiale de la [Phase 1](../03_phase_1_experience_pilote/02_decouverte_fuite_spatiale.md).
> Le jeu de données exploité à partir de la Phase 2 est produit par le chapitre [`06`](06_decoupage_train_val_test.md), qui rejoue cet export **trois fois**, une par volet géographique. Le présent chapitre reste la référence des paramètres de tuilage, repris à l'identique.

Cette étape découpe le raster géoréférencé et la couche de masques vectoriels nettoyée en vignettes régulières directement consommables par les dataloaders PyTorch / torchvision / transformers.

---

## 1. Paramétrage de l'outil d'export ArcGIS

- **Outil** : `Geoprocessing → Export Training Data For Deep Learning` (Image Analyst Tools).
- **Raster d'entrée (`Input Raster`)** : `ortho_gpso_brut_Clip`.
- **Dossier de sortie (`Output Folder`)** : `tiles/`.
- **Classe d'entités vectorielles (`Input Feature Class`)** : `passages_pietons_all`.
- **Polygones de masquage (`Input Mask Polygons`)** : `quartiers_annotes`.
- **Format d'image (`Image Format`)** : `TIFF` (avec fichiers de géoréférencement `.tfw`).
- **Format de métadonnées (`Metadata Format`)** : `Classified_Tiles`.
- **Dimensions de tuile (`Tile Size X / Y`)** : $512 \times 512$ pixels ($102,4\text{ m} \times 102,4\text{ m}$).
- **Pas d'échantillonnage (`Stride X / Y`)** : $256 \times 256$ pixels ($51,2\text{ m} \times 51,2\text{ m}$).
- **Filtre de tuiles vides (`Output No Feature Tiles`)** : `ONLY_TILES_WITH_FEATURES` (conserve uniquement les tuiles contenant au moins un pixel de passage piéton).
- **Système de référence (`Reference System`)** : `MAP_SPACE` (coordonnées métriques projetées Lambert-93).
- **Mode de découpe (`Crop Mode`)** : `FIXED_SIZE`.

### ⏱️ Logs d'exécution ArcGIS :
```
Start Time: jeudi 20 août 2026 13:14:49
Working with cell size 0,200000 meter.
Distributing operation across 11 parallel instances.
NumTilesWritten = 197, NextTileIndex = 650
NumTilesWritten = 162, NextTileIndex = 1227
NumTilesWritten = 101, NextTileIndex = 1625
NumTilesWritten = 7, NextTileIndex = 1662
NumTilesWritten = 27, NextTileIndex = 1871
NumTilesWritten = 88, NextTileIndex = 2487
Succeeded at jeudi 20 août 2026 13:17:00 (Elapsed Time: 2 minutes 11 seconds)
```

---

## 2. Structure et organisation du dossier `tiles/`

Le dossier `tiles/` à la racine du projet contient 582 paires d'images et de masques parfaitement synchronisées :

```
tiles/
├── images/                         # 582 fichiers .tif RVB + 582 fichiers .tfw
│   ├── 000000000.tif               # Image 512x512 uint8
│   ├── 000000000.tfw               # Coordonnées Lambert-93 du coin haut-gauche
│   └── ...
├── labels/                         # 582 masques .tif + 582 fichiers .tfw
│   ├── 000000000.tif               # Masque 512x512 (valeurs 0 ou 1)
│   ├── 000000000.tfw
│   └── ...
├── map.txt                         # Table d'appariement image -> label
├── stats.txt                       # Statistiques globales de classes générées par ArcGIS
├── esri_model_definition.emd       # Définition de modèle ESRI et métadonnées d'export
└── esri_accumulated_stats.json     # Histogrammes spectraux et statistiques de bandes
```

---

## 3. Statistiques descriptives du jeu de données

- **Nombre total de tuiles** : **582 vignettes**.
- **Résolution spatiale** : $0,20\text{ m/pixel}$ ($102,4\text{ m}$ de côté par vignette).
- **Surface totale imagée** : $\approx 6,1\text{ km}^2$ (avec chevauchement).
- **Proportion de pixels positifs (Passages piétons)** : **0,6725 %** (soit $\approx 1\ 026\ 039\text{ pixels}$).
- **Nombre de passages piétons distincts dans les tuiles** : **1 691 occurrences** (un passage peut être visible sur plusieurs tuiles adjacentes en raison du stride 256 px).
- **Nombre moyen de passages par tuile** : 2,91 (min = 1, max = 13).
