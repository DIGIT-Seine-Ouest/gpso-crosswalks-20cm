# Bilan — U-Net ResNet34 (seed 0)

> ⚠️ **DRAFT** — 25 août 2026. Une seule graine. Volet test jamais ouvert.

## Configuration

Fine-tuning complet, **41,22 M paramètres entraînés**. Loss `combo`, lr 1e-4, 30 époques, batch 2, `mps`, ~2 h 55.
Checkpoint retenu : **époque 20 / 30**. Empreintes données identiques au run DINOv3.

## Résultats — volet val (88 tuiles, 189 occurrences, seuil 0,5)

| | |
|---|---|
| **IoU** | **0,7459** — IC 95 % [0,7118 – 0,7743] |
| Dice / Précision / Rappel | 0,8545 / 0,8685 / 0,8408 |
| **Détection d'objet** | **92,6 %** (175/189) |
| IoU par tuile (p10 / méd / p90) | 0,451 / 0,750 / 0,868 |
| Pixels erronés | 34 954 — dont **68,4 % au contour** |
| Accuracy globale | 0,9985 *(baseline tout-fond : 0,9947 — métrique inutile)* |

## L'essentiel

- **Convergé dès l'époque 8** (0,7192). Le pic de l'époque 20 ne gagne que 0,027 sur ce plateau → **le 0,7459 est un peu flatteur**, l'écart est dans le bruit résiduel de la courbe. 30 époques sont surdimensionnées.
- **Sous-segmentation** : le terme d'erreur dominant est `FN contour` à 42,6 % — bandes dessinées trop maigres, de l'ordre de ±1 px = 20 cm terrain. Sans conséquence sur le centroïde.
- **Erreurs « loin » divisées par plus de deux** vs DINOv3 : 11 044 px (31,6 %) contre 24 596 px (43,8 %). Ce sont les seules qui produiraient des points parasites en SIG.
- **Aucun passage complet manqué** : les 14 objets ratés font tous moins de 617 px, soit des fragments de bord de tuile. DINOv3 laisse passer 7 objets pleins sur ses 24 manques.
- **Seuil sans effet** : 0,7329 à 0,7485 sur tout l'intervalle. Rien à ajuster.
- **TTA D4 à écarter** : dégrade tout (IoU 0,7434, détection 91,0 %) pour 8× le coût.

## Limites

1. Une seule graine, aucune dispersion mesurée — d'autant plus gênant que le pic est mal séparé du plateau.
2. **Régime asymétrique** : fine-tuning complet contre backbone gelé. Conclusion défendable = « U-Net en fine-tuning complet > DINOv3 gelé », pas « U-Net > DINOv3 ». L'ablation `--freeze-encoder` trancherait.
3. Erreur de centroïde et MAE de comptage non mesurées (supposent la vectorisation Lambert-93).
4. Volet test jamais ouvert : tous ces chiffres sont de la validation.

## Conclusion

Devance DINOv3 sur toutes les métriques mesurées : **+0,108 d'IoU (+16,8 %)**, **+5,3 pts de détection**. Les IC 95 % ne se recouvrent pas — l'écart n'est pas du bruit.

Dépasse aussi le 0,7252 du pilote de Phase 1, qui était obtenu sur données fuitées à 84 %.

---

*Détails : [`metrics_val.json`](../../datasets/runs/unet-resnet34/metrics_val.json) · [figures](../../datasets/runs/unet-resnet34/figures/) · [match qualitatif tuile par tuile](../../datasets/runs/comparatif_dinov3_vs_unet-resnet34_val.html)*
