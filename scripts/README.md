# `scripts/` — entrainer, evaluer, tracer

Le paquet Python du depot. Trois points d'entree, une option `--model`, et le
meme protocole pour toutes les architectures.

---

## Demarrage rapide

```bash
uv sync --extra tracking                    # torch, transformers, mlflow
python -m scripts.hf_sync pull              # recupere donnees et runs du bucket

# une epoque pour verifier que tout repond
python -m scripts.train --model dinov3 --device mps --epochs 1 --track
```

L'en-tete affiche ce qu'il faut lire avant de lancer un vrai run :

```
[data]  374 train / 88 val (dossiers tiles_*)
[data]  empreintes train 9fa11b40eeb0 / val 03d992fffbd5
[model] DINOv3 ViT-L/16 SAT-493M gele + tete conv | tete 2.36 M entrainables, backbone gele 304 M
[calc]  mps (Apple Silicon, memoire unifiee) | precision mixte fp16 : non
[track] mlflow | experience gpso-crosswalks | run 7f3a... | file://.../datasets/mlruns
```

Si `[track]` affiche `desactive`, le tracage a echoue mais l'entrainement
continue : c'est voulu, jamais l'inverse.

Puis le vrai run, et les courbes :

```bash
python -m scripts.train --model dinov3 --device mps --workers 2 --track
mlflow ui --backend-store-uri file://$(pwd)/datasets/mlruns
```

Le `--backend-store-uri` n'est pas optionnel : `mlflow ui` cherche `./mlruns`
alors que le store vit sous `datasets/`. Sans lui, l'interface est vide.

---

## Les trois commandes

```bash
python -m scripts.train     --model dinov3
python -m scripts.evaluate  --model dinov3 --checkpoint datasets/runs/dinov3/seed0.pt
python -m scripts.visualize --model dinov3 --checkpoint datasets/runs/dinov3/seed0.pt
```

Architectures disponibles : `dinov3`, `unet-resnet34`, `unet-resnet50`.

### Options de `train.py`

| Option | Defaut | |
|---|---|---|
| `--model` | *requis* | l'architecture |
| `--seed` | `0` | relancer avec 0, 1, 2 donne la dispersion |
| `--epochs` | `30` | |
| `--batch-size` | `2` | **ne pas changer** entre modeles compares |
| `--lr` | celui du modele | `1e-3` pour dinov3, `1e-4` pour un U-Net |
| `--recipe` | `protocole` | ou `arcgis`, cf. plus bas |
| `--device` | le plus rapide present | `cuda`, `mps` ou `cpu` |
| `--amp` / `--no-amp` | fp16 sur cuda | force ou coupe la precision mixte |
| `--workers` | `0` | mettre `2` sur Colab et sur Mac |
| `--track` | non | trace le run dans MLflow |
| `--log-checkpoint` | non | joint le `.pt` aux artefacts du run |

Les sorties atterrissent dans `datasets/runs/{modele}/` : `seed0.pt` (le meilleur
checkpoint sur val) et `history_seed0.json` (la courbe, **reecrite a chaque
epoque** — un run interrompu laisse un resultat exploitable).

### Le protocole multi-seed

```bash
for s in 0 1 2; do python -m scripts.train --model dinov3        --seed $s --track; done
for s in 0 1 2; do python -m scripts.train --model unet-resnet34 --seed $s --track; done
```

Puis, **une seule fois**, apres que la meilleure epoque a ete choisie sur val :

```bash
python -m scripts.evaluate --model dinov3 --checkpoint datasets/runs/dinov3/seed0.pt \
                           --subset test --track
```

Ouvrir le volet test avant d'avoir fige le checkpoint revient a s'entrainer
dessus. `train.py` ne le touche jamais.

### La baseline ArcGIS

```bash
python -m scripts.evaluate --model unet-resnet34 --weights resnet_unet.pth
```

Le rapport porte alors une reserve explicite : ces poids ont ete entraines par
ArcGIS sur son propre decoupage aleatoire, inconnu, et leur score n'est pas
comparable a celui d'un modele entraine ici.

---

## Ou tourne l'entrainement

L'ordre est `cuda > mps > cpu`, et `--device` force un choix. Une demande
indisponible est une erreur explicite, jamais un repli silencieux : decouvrir
apres deux heures qu'un run a tourne sur le processeur est le genre de surprise
que ce depot evite.

La precision mixte fp16 est active par defaut sur CUDA. Sur MPS elle reste
facultative (`--amp`), certains noyaux etant encore instables selon la version de
torch ; le `GradScaler`, propre a CUDA, y est desactive.

Le peripherique ne change ni les tuiles, ni la loss, ni les metriques : **un IoU
obtenu sur MPS et un IoU obtenu sur T4 se comparent. Les temps par epoque, non.**
D'ou l'enregistrement du peripherique dans le checkpoint et l'historique.

### Laisser tourner la nuit

```bash
nohup caffeinate -is python -u -m scripts.train \
      --model dinov3 --device mps --workers 2 --track \
      > datasets/runs/seed0.log 2>&1 &
tail -f datasets/runs/seed0.log
```

`python -u` sinon la sortie reste en tampon dans le fichier pendant des heures.
`caffeinate -is` empeche la veille, mais ne survit pas a la fermeture du capot :
laisser l'ecran ouvert et la machine sur secteur. Un `Ctrl-C` clot le run en
`KILLED` avec son historique et son meilleur checkpoint deja ecrits.

---

## Suivi des experiences

Optionnel. Sans `--track`, ou sans mlflow installe, le tracage est muet et
l'entrainement rigoureusement identique : un Colab minimal n'a rien de plus a
installer. **Une erreur de tracage ne fait jamais echouer un entrainement** —
toute exception le desactive, l'annonce une fois, et la boucle continue.

| | contenu |
|---|---|
| parametres | modele, recette, graine, epoques, batch, lr, loss, peripherique, precision, nombre de tuiles, **empreintes des volets** |
| metriques | par epoque : train_loss, valid_loss, accuracy, dice, IoU, duree ; en fin : meilleur IoU et son epoque |
| artefacts | l'historique JSON, le rapport d'evaluation, le checkpoint sur `--log-checkpoint` |
| tags | commit git et proprete de l'arbre, machine, plateforme, version de torch |

L'**empreinte** est un SHA-256 des noms de tuiles et de leurs tailles
(`dataset.fingerprint`). Deux runs aux memes hyperparametres mais aux empreintes
differentes n'ont pas vu les memes donnees et ne se comparent pas. C'est ce
parametre qui aurait revele immediatement les 23 tuiles ecrasees a l'export
(cf. [`datasets/README.md`](../datasets/README.md)).

### Changer de machine

Tout vit sous `datasets/` — les tuiles, `runs/`, `mlruns/` — et `datasets/` est
le miroir local du bucket
[`gpso-crosswalks-20cm`](https://huggingface.co/buckets/mandresyandri/gpso-crosswalks-20cm).
Passer du Mac a Colab tient donc en deux commandes :

```bash
python -m scripts.hf_sync pull      # avant d'entrainer
python -m scripts.hf_sync push      # apres
```

`--dry-run` montre le plan sans rien transferer. `--no-weights` laisse les
checkpoints au sol quand seules les courbes doivent voyager (un `.pt` de U-Net
pese 160 Mo, celui de dinov3 en pese 9). `--delete` elague la destination, a
n'utiliser qu'apres un `--dry-run`.

Le script pose toujours `--ignore-times`, et ce n'est pas un detail : le bucket
ne renvoie pas de date de modification — il repond `1970-01-01` pour chaque objet
— si bien que la comparaison par defaut, qui croise taille et date, classe les
2 837 fichiers en « local plus recent » et reenvoie tout le jeu a chaque `push`.

---

## Ou sont les donnees

Le decoupage train / val / test est **spatial et fige a l'export** : un dossier
par volet, produit par le chapitre
[`06`](../cahier_recherche/02_acquisition_de_donnees/06_decoupage_train_val_test.md)
du cahier de recherche. Il n'y a pas de fichier de split a charger.

```
datasets/
├── tiles_train/    374 tuiles — Boulogne-Billancourt + Sevres
├── tiles_val/       88 tuiles — Meudon-sur-Seine
├── tiles_test/     101 tuiles — Vanves
├── runs/           checkpoints, historiques, rapports
└── mlruns/         store MLflow
```

Le train en porte 374 et non les 397 qu'annoncent `map.txt` et les metadonnees
ArcGIS : 23 tuiles se sont ecrasees a l'export. Le detail est dans
[`datasets/README.md`](../datasets/README.md).

Quatre variables d'environnement deplacent les chemins par defaut :

| | |
|---|---|
| `GPSO_DATA` | racine des jeux (defaut `datasets/`) |
| `GPSO_RUNS` | checkpoints et historiques |
| `GPSO_MLRUNS` | store MLflow |
| `GPSO_BUCKET` | bucket cible de `hf_sync` |

L'option `--split` charge un `split.json` a l'ancienne, sur le jeu unique du
pilote. Elle ne sert qu'a rejouer la Phase 1, dont le decoupage aleatoire
fuyait : **elle ne produit pas un resultat publiable.**

### Attention aux masques

Les masques de `labels/` sont ecrits par ArcGIS en **entiers non signes sur
16 bits**, alors que les images sont en 8 bits sur trois bandes. Les valeurs
utiles restent `{0, 1}`. Un lecteur qui suppose du 8 bits partout obtient un
masque etire d'un facteur 2 et decale, **sans lever la moindre exception** —
`dataset.TileSet` verifie donc les valeurs a la premiere lecture et s'arrete net
si elles sortent de `{0, 1}`.

---

## Recettes d'entrainement

| `--recipe` | Optimiseur | Loss | Usage |
|---|---|---|---|
| `protocole` *(defaut)* | AdamW + CosineAnnealingLR, un seul lr | 0,5 CE ponderee + 0,5 Dice | la recette de reference, celle qui rend les modeles comparables |
| `arcgis` | AdamW + OneCycleLR, lr discriminants entre `lr/10` et `lr` | CrossEntropy nue | reconstitution de la recette ArcGIS, pour verifier qu'on retombe sur la baseline |

---

## Sous le capot

Un principe : **aucun module transverse ne connait un modele en particulier.**
Ajouter une architecture ne demande de toucher ni a `train.py`, ni a
`evaluate.py`, ni a `visualize.py`.

```
scripts/
├── config.py        chemins, normalisations, geometrie des tuiles
├── dataset.py       lecture des tuiles et des masques, augmentation D4, empreintes
├── device.py        choix cuda / mps / cpu et precision mixte
├── metrics.py       metriques pixel et objet, TTA D4, rapport
├── tracking.py      tracage MLflow, optionnel et jamais bloquant
├── hf_sync.py       synchronisation de datasets/ avec le bucket Hugging Face
├── models/
│   ├── base.py      SegModel : le contrat que respecte toute architecture
│   ├── __init__.py  registre, import differe des dependances
│   ├── dinov3.py    ViT-L/16 SAT-493M gele + tete conv
│   └── unet.py      U-Net a encodeur ResNet (34, 50, ...)
├── train.py         point d'entree unique  --model
├── evaluate.py      point d'entree unique  --model
└── visualize.py     point d'entree unique  --model
```

### Ce qui doit rester identique entre modeles

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

### Ajouter une architecture

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
