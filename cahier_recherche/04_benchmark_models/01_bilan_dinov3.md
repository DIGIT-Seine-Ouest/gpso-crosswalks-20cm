# Bilan — DINOv3 ViT-L/16 gelé (seed 0)

> ⚠️ **DRAFT** — 25 août 2026. Une seule graine. Volet test jamais ouvert.

## Configuration

Backbone **DINOv3 ViT-L/16 SAT-493M gelé** + tête convolutive — **2,36 M paramètres entraînés** sur 304 M gelés.
Loss `combo`, lr 1e-3, 30 époques, batch 2, `mps`, ~3 h 54 (moyenne 7 min 49 / époque).
Checkpoint retenu : **époque 18 / 30**. 374 tuiles d'entraînement (et non 397 : 23 écrasées à l'export).

## Résultats — volet val (88 tuiles, 189 occurrences, seuil 0,5)

| | |
|---|---|
| **IoU** | **0,6384** — IC 95 % [0,6127 – 0,6637] |
| Dice / Précision / Rappel | 0,7793 / 0,7487 / 0,8125 |
| **Détection d'objet** | **87,3 %** (165/189) |
| IoU par tuile (p10 / méd / p90) | 0,469 / 0,669 / 0,758 |
| Pixels erronés | 56 163 — dont **56,2 % au contour** |
| Accuracy globale | 0,9976 *(baseline tout-fond : 0,9947 — métrique inutile)* |

## L'essentiel

- **0,725 → 0,638 n'est pas une régression, c'est une correction.** Le pilote de Phase 1 était contaminé par 84 % de sol partagé entre train et val (stride 256 px). Le 0,6384 est le premier chiffre du projet exempt de fuite.
- **La métrique métier a bien moins bougé que la métrique pixel** : 88,1 % → 87,3 % de détection (−0,8 pt) contre −12 % relatifs sur l'IoU. La fuite gonflait surtout la finesse du contour, pas la capacité à voir un passage. *(Comparaison non stricte : les deux volets de validation diffèrent.)*
- **Il ne hallucine pas, il déborde.** 82 % de ses faux positifs sont à moins de 8 px d'un passage réel — un halo, pas un objet inventé. Ses amas isolés sont comparables à ceux du U-Net (14 contre 11). Sa précision plus faible (0,749) tient à la grille de patchs 16 px du ViT-L, pas à la sémantique du backbone.
- **Rarement catastrophique, jamais net** : une seule tuile au-dessus de 0,80 d'IoU, contre 31 pour le U-Net.
- **Son avantage propre : le bas contraste.** C'est le seul terrain où il bat le U-Net, qui rend une prédiction vide en zone d'ombre. Robustesse radiométrique du pré-entraînement SAT-493M — pas anecdotique sur un territoire d'ombres de bâti.
- **TTA D4 ambivalente** : +0,0195 d'IoU (0,6579) mais **détection en baisse**, 87,3 → 85,7 %. Gain pixel payé en objets perdus.

## Limites

1. Une seule graine, aucune dispersion mesurée.
2. Erreur de centroïde et MAE de comptage non mesurées (supposent la vectorisation Lambert-93).
3. Volet test jamais ouvert : tous ces chiffres sont de la validation.
4. ⚠️ **Décomptes d'objets à instruire.** Trois sources incompatibles pour le volet val : 189 occurrences par tuile (`metrics_val.json`), **55 passages distincts** après recollage Lambert-93, et 106 dans les métadonnées ArcGIS du chapitre [`02/06`](../02_acquisition_de_donnees/06_decoupage_train_val_test.md). Le « 59 » qui figure encore dans `diagnostic_val.html` est un comptage manuel non reproductible, à corriger.

## Conclusion

Premier chiffre honnête du projet, obtenu en n'entraînant que 2,36 M de paramètres — le backbone gelé fait presque tout le travail. Mais **devancé par le U-Net ResNet34** sous le même protocole (0,6384 contre 0,7459 ; 87,3 % contre 92,6 % de détection), avec des IC 95 % qui ne se recouvrent pas.

Reste le modèle le plus économique en mémoire (18 % de VRAM contre 96 % pour le U-Net en Phase 1) et le plus stable à l'entraînement.

---

*Détails : [`metrics_val.json`](../../datasets/runs/dinov3/metrics_val.json) · [figures](../../datasets/runs/dinov3/figures/) · [match qualitatif tuile par tuile](../../datasets/runs/comparatif_dinov3_vs_unet-resnet34_val.html)*
