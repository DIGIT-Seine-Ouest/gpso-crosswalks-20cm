# scripts/ — références d'entraînement et d'évaluation

Scripts repris de l'expérience pilote et conservés ici comme **base d'inspiration** pour
la phase 2. Ils n'ont pas été réécrits : c'est le code tel qu'il a produit les résultats
du `RESULTATS.md` de l'archive. Le reste du pilote (poids, métriques, figures, journal)
est archivé dans `legacy.zip` à la racine.

```
scripts/
├── eval_common.py          <- Socle d'évaluation partagé (métriques objet, TTA, dataset)
├── dinov3/
│   ├── train_dinov3.py     <- Entraînement de la tête sur DINOv3 ViT-L/16 SAT-493M gelé
│   ├── evaluate.py         <- Évaluation d'un checkpoint via eval_common
│   ├── train_dino_compare.py  <- Variantes d'entraînement pour l'ablation
│   └── visualize.py        <- Figures qualitatives et cartes d'erreurs
└── resnet_unet/
    ├── train_unet.py       <- Baseline U-Net ResNet34 (recettes « protocole » et « arcgis »)
    └── evaluate_unet.py    <- Évaluation de la baseline via eval_common
```

## Résolution des chemins

`eval_common.py` et `train_dinov3.py` déduisent la racine du dépôt de leur propre
emplacement. Placés ici, ils pointent correctement sur `tiles/` — c'était cassé dans
`legacy/models/` depuis l'archivage.

En revanche le **split par défaut** (`split.json` à la racine) n'existe pas : les splits
sont dans `legacy.zip`. Il faut les extraire une fois, puis les passer explicitement.

```bash
# Extraction du split spatialement disjoint (11 Ko)
unzip -j legacy.zip 'legacy/split_v2.json' -d .

# Entraînement DINOv3
python scripts/dinov3/train_dinov3.py --split split_v2.json --epochs 30

# Baseline U-Net
python scripts/resnet_unet/train_unet.py --split split_v2.json --epochs 30

# Évaluation d'un checkpoint archivé (à extraire lui aussi de legacy.zip)
unzip -j legacy.zip 'legacy/models/dinov3/best_dinov3_head.pt' -d .
python scripts/dinov3/evaluate.py --split split_v2.json \
    --checkpoint best_dinov3_head.pt --subset test
```

Deux variables d'environnement court-circuitent ces défauts : `GPSO_TILES` et `GPSO_SPLIT`.

## Ce qui n'a pas été copié

Les **poids** (`best_dinov3_head.pt`, `resnet_unet.pth`, `.dlpk`), les **métriques** et les
**figures** sont dans `legacy.zip`. Idem pour les scripts transverses `benchmark.py`,
`compare_models.py` et `make_split.py`, qui n'appartiennent à aucun des deux modèles en
propre. Pour explorer l'archive : `unzip -l legacy.zip`.
