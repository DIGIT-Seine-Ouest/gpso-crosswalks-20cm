# `scripts/` — entrainement, evaluation, figures

Paquet Python du depot. Un principe : **aucun module transverse ne connait un
modele en particulier**. Ajouter une architecture ne demande de toucher ni a
`train.py`, ni a `evaluate.py`, ni a `visualize.py`.

```
scripts/
├── config.py        chemins, normalisations, geometrie des tuiles
├── dataset.py       lecture des tuiles et des masques, augmentation D4
├── metrics.py       metriques pixel et objet, TTA D4, rapport
├── models/
│   ├── base.py      SegModel : le contrat que respecte toute architecture
│   ├── __init__.py  registre, import differe des dependances
│   ├── dinov3.py    ViT-L/16 SAT-493M gele + tete conv
│   └── unet.py      U-Net a encodeur ResNet (34, 50, ...)
├── train.py         point d'entree unique  --model
├── evaluate.py      point d'entree unique  --model
└── visualize.py     point d'entree unique  --model
```

## Utilisation

Les points d'entree se lancent comme des modules, depuis la racine du depot :

```bash
python -m scripts.train    --model dinov3
python -m scripts.evaluate --model dinov3 --checkpoint runs/dinov3/seed0.pt
python -m scripts.visualize --model dinov3 --checkpoint runs/dinov3/seed0.pt
```

Les trois graines du protocole multi-seed :

```bash
for s in 0 1 2; do python -m scripts.train --model dinov3        --seed $s; done
for s in 0 1 2; do python -m scripts.train --model unet-resnet34 --seed $s; done
```

Evaluation finale sur le volet test, **une seule fois**, apres que la meilleure
epoque a ete choisie sur val :

```bash
python -m scripts.evaluate --model dinov3 --checkpoint runs/dinov3/seed0.pt --subset test
```

Baseline ArcGIS, dont les poids existent mais pas le decoupage :

```bash
python -m scripts.evaluate --model unet-resnet34 --weights resnet_unet.pth
```

Le rapport porte alors une reserve explicite : ces poids ont ete entraines par
ArcGIS sur son propre decoupage aleatoire, inconnu, et leur score n'est pas
comparable a celui d'un modele entraine ici.

## Ou sont les donnees

Le decoupage train / val / test est **spatial et fige a l'export** : un dossier
par volet, produit par le chapitre
[`06`](../cahier_recherche/02_acquisition_de_donnees/06_decoupage_train_val_test.md)
du cahier de recherche. Il n'y a pas de fichier de split a charger.

```
datasets/
├── tiles_train/    397 tuiles — Boulogne-Billancourt + Sevres
├── tiles_val/       88 tuiles — Meudon-sur-Seine
└── tiles_test/     101 tuiles — Vanves
```

Chaque dossier contient `images/` et `labels/`. Trois variables d'environnement
deplacent les chemins par defaut : `GPSO_DATA` (racine des jeux), `GPSO_TILES`
(jeu du pilote), `GPSO_RUNS` (sorties).

L'option `--split` charge un `split.json` a l'ancienne, sur le jeu unique du
pilote. Elle ne sert qu'a rejouer la Phase 1, dont le decoupage aleatoire
fuyait : elle ne produit pas un resultat publiable.

## Ajouter une architecture

Deux etapes, et rien d'autre dans le depot.

**1.** Un module `models/mon_modele.py` qui expose une constante `NORMALIZATION`
et une fonction `build(**kwargs)` renvoyant une instance de `SegModel` :

```python
from .. import config
from .base import SegModel

NORMALIZATION = config.IMAGENET

class MonModele(SegModel):
    NAME = "mon-modele"
    DESCRIPTION = "..."
    NORMALIZATION = NORMALIZATION
    DEFAULT_LR = 1e-4

    def forward(self, x):
        ...   # (B, 3, H, W) -> logits (B, 2, H, W), classe 1 = passage pieton

def build(**kwargs):
    return MonModele(**kwargs)
```

**2.** Une ligne dans `_REGISTRY`, dans `models/__init__.py`.

Le contrat `SegModel` tient en quatre points, dont trois ont un comportement par
defaut utilisable tel quel :

| Membre | Role | Defaut |
|---|---|---|
| `forward(x)` | logits `(B, 2, H, W)` | a ecrire |
| `NORMALIZATION` | pretraitement d'entree du pre-entrainement | ImageNet |
| `trainable_parameters()` | ce que voit l'optimiseur | tout ce qui a `requires_grad` |
| `trainable_state_dict()` | ce qu'on sauvegarde | le modele entier |

Un backbone gele redefinit les deux derniers, pour ne pas ecrire des centaines
de mega-octets de poids figes a chaque epoque : `dinov3.py` en donne l'exemple.

Un U-Net a encodeur ResNet50 est deja declare (`--model unet-resnet50`) : il
reutilise `models/unet.py` avec un autre encodeur, sans une ligne de code
supplementaire.

## Ce qui doit rester identique entre modeles

C'est la condition pour qu'un ecart dans le tableau comparatif vienne des
modeles et non du protocole. Ces elements vivent dans les modules transverses,
hors d'atteinte d'une architecture :

- les tuiles et le decoupage spatial (`dataset.py`) ;
- l'augmentation D4, les 8 symetries du carre (`dataset.py`) ;
- la loss, le nombre d'epoques, la selection du checkpoint sur val (`train.py`) ;
- toutes les metriques (`metrics.py`).

Trois choses restent legitimement propres a chaque modele, parce que les imposer
fausserait la comparaison plutot que de la garantir : la normalisation d'entree
(chaque reseau recoit le pretraitement de son pre-entrainement), le taux
d'apprentissage par defaut, et les groupes de parametres pour les taux
discriminants.

## Recettes d'entrainement

| `--recipe` | Optimiseur | Loss | Usage |
|---|---|---|---|
| `protocole` *(defaut)* | AdamW + CosineAnnealingLR, un seul lr | 0,5 CE ponderee + 0,5 Dice | la recette de reference, celle qui rend les modeles comparables |
| `arcgis` | AdamW + OneCycleLR, lr discriminants entre `lr/10` et `lr` | CrossEntropy nue | reconstitution de la recette ArcGIS, pour verifier qu'on retombe sur la baseline |

## Note sur les masques

Les masques de `labels/` sont ecrits par ArcGIS en **entiers non signes sur
16 bits**, alors que les images sont en 8 bits sur trois bandes. Les valeurs
utiles restent `{0, 1}`. Un lecteur qui suppose du 8 bits partout obtient un
masque etire d'un facteur 2 et decale, **sans lever la moindre exception** —
`dataset.TileSet` verifie donc les valeurs a la premiere lecture et s'arrete net
si elles sortent de `{0, 1}`.
