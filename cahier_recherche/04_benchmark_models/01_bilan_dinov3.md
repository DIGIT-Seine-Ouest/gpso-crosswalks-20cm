# 01 — Bilan du run DINOv3 gelé sous protocole étanche

> ⚠️ **DRAFT** — rédigé le 25 août 2026 à partir d'un run unique (seed 0). Chiffres relus dans `datasets/runs/dinov3/`, non encore confirmés par une seconde graine.
>
> **Statut** : run achevé (30/30 époques), évalué sur le volet **validation** uniquement.
> **Sources** : `datasets/runs/dinov3/metrics_val.json`, `history_seed0.json`, `datasets/runs/seed0.log`, MLflow `gpso-crosswalks` / `dinov3_protocole_seed0`.

---

## 🎯 Ce que ce run mesure — et ce qu'il ne mesure pas

Il mesure ce que fait un DINOv3 gelé, entraîné sur Boulogne-Billancourt et Sèvres, lorsqu'on lui présente **Meudon-sur-Seine, une commune dont il n'a jamais vu un mètre carré**. C'est la première fois dans ce projet qu'un chiffre décrit cette situation.

Il ne mesure **pas** la performance sur le territoire GPSO complet, ni la performance finale du projet :

- le volet **test (101 tuiles, Vanves) n'a pas été ouvert** et ne le sera qu'une fois, l'architecture et l'époque figées ;
- les métriques cardinales du [cadrage](../01_cadrage_et_metriques/01_importance_des_metriques.md) — erreur de centroïde, MAE de comptage — **n'ont encore aucun chiffre**, elles supposent le recollage en Lambert-93 ;
- une seule graine a tourné.

---

## ⚙️ Configuration exacte du run

| | |
|---|---|
| **Modèle** | DINOv3 ViT-L/16 SAT-493M **gelé** + tête convolutive 2 couches |
| **Paramètres** | 2,36 M entraînables / 304 M gelés (**0,8 %** du réseau reçoit un gradient) |
| **Recette** | `protocole` — AdamW + CosineAnnealingLR, un seul lr |
| **Loss** | `combo` : 0,5 × CrossEntropy pondérée (`pos_weight` 20) + 0,5 × Dice |
| **Hyperparamètres** | 30 époques, batch 2, lr $1\times10^{-3}$, seed 0 |
| **Données** | 374 tuiles train (187 lots) / 88 tuiles val (44 lots) |
| **Empreintes des volets** | train `9fa11b40eeb0` / val `03d992fffbd5` |
| **Calcul** | `mps` (Apple Silicon, mémoire unifiée), précision mixte fp16 **désactivée**, 2 workers |
| **Durée** | 7 min 13 s à 8 min 31 s par époque (moyenne 7 min 49 s), **~3 h 54 au total** (24 août 22 h 05 → 25 août 01 h 59) |
| **Checkpoint retenu** | `datasets/runs/dinov3/seed0.pt` — **époque 18**, IoU val 0,6384 |
| **Traçabilité** | MLflow, expérience `gpso-crosswalks`, run `dinov3_protocole_seed0` ; commit `154f0e7`, arbre propre ; modèle inscrit au registre sous **`gpso-dinov3` version 1** |

> ⚠️ **374 tuiles d'entraînement, et non les 397 annoncées** par le chapitre [`02/06`](../02_acquisition_de_donnees/06_decoupage_train_val_test.md) : 23 tuiles se sont écrasées à l'export ArcGIS (voir [`datasets/README.md`](../../datasets/README.md)). L'empreinte `9fa11b40eeb0` est ce qui fait foi pour comparer deux runs : deux entraînements aux mêmes hyperparamètres mais aux empreintes différentes n'ont pas vu les mêmes données.

---

## 📉 La courbe des 30 époques

![Courbes d'entraînement DINOv3](../../datasets/runs/dinov3/figures/training.png)
*Figure 1 : IoU et Dice sur le volet de validation, 30 époques. Point rouge : l'époque 18 retenue.*

Trois choses se lisent sur cette courbe :

1. **Le backbone gelé travaille dès l'époque 0** : IoU 0,5483 après un seul passage sur les données. La tête n'a que 2,36 M de paramètres à ajuster — les 29 époques suivantes ne rapportent que **+0,09 d'IoU**. L'essentiel de la performance vient des représentations pré-entraînées, pas de l'apprentissage sur GPSO.
2. **Le plateau est atteint vers l'époque 11** (IoU 0,6362) et n'est plus quitté. Les époques 11 à 29 oscillent entre 0,600 et 0,638 : l'amplitude de bruit d'une époque à l'autre est du même ordre que les écarts qu'on cherche à mesurer.
3. **Après l'époque 18, la perte d'entraînement continue de baisser (0,1048 → 0,0932) sans que l'IoU de validation suive.** C'est la signature d'un début de sur-apprentissage, léger — la tête est trop petite pour dériver franchement.

L'époque 18 n'est pas seulement le maximum d'IoU : c'est aussi le **minimum de perte de validation** (0,1695). Les deux critères désignent la même époque, ce qui rend la sélection moins fragile que si elle avait reposé sur l'IoU seul.

---

## 📊 Résultats sur le volet validation

**88 tuiles de Meudon-sur-Seine, 189 occurrences d'objets, seuil de décision 0,5.**

| Métrique pixel | Valeur | IC 95 % |
|---|---|---|
| **IoU passage piéton** | **0,6384** | [0,6127 – 0,6637] |
| Dice / F1 passage piéton | 0,7793 | |
| Précision pixel | 0,7487 | |
| Rappel pixel | 0,8125 | |
| IoU fond | 0,9976 | |

Le modèle **prédit plus large que la vérité terrain** : rappel (0,812) supérieur à précision (0,749), et 59,3 % des pixels erronés sont des faux positifs contre 40,7 % de faux négatifs. Il rate peu de surface, il en ajoute.

### Dispersion d'une tuile à l'autre

| | IoU |
|---|---|
| Médiane par tuile | 0,6686 |
| Décile bas (p10) | 0,4692 |
| Décile haut (p90) | 0,7580 |

La médiane par tuile (0,669) dépasse l'IoU global (0,638) : les tuiles qui portent beaucoup de surface de passage piéton tirent le score global vers le bas. Réparties par qualité, les 88 tuiles donnent **31 bonnes (IoU ≥ 0,70), 38 moyennes, 19 mauvaises (IoU < 0,55)** — dont 4 à IoU exactement nul, toutes analysées plus bas.

---

## 🚫 L'accuracy globale de 0,9976 ne dit rien

| | Pixels bien classés | Pixels erronés sur une tuile de 262 144 px |
|---|---|---|
| **Modèle DINOv3** | 99,7565 % | ~638 |
| **Baseline aveugle « tout est du fond »** | 99,4710 % | ~1 387 |

Un modèle qui prédirait une image entièrement noire obtiendrait **99,47 %** sur ce volet. Le modèle entraîné obtient 99,76 %. **L'écart entier tient dans les 0,29 point** que sépare l'inutile du fonctionnel.

C'est exactement le piège documenté au [chapitre 01 du cadrage](../01_cadrage_et_metriques/01_importance_des_metriques.md) : sur une classe qui occupe 0,53 % des pixels de ce volet, l'accuracy mesure la rareté de la classe, pas la qualité du modèle. Elle est calculée et journalisée par `metrics.py` **uniquement pour pouvoir être rapprochée de sa baseline** — jamais pour être citée seule.

---

## 🔍 Où sont les erreurs : 56 % sur le contour

![Décomposition des erreurs et sensibilité au seuil](../../datasets/runs/dinov3/figures/errors.png)
*Figure 2 : à gauche, les 56 163 pixels erronés répartis en quatre familles ; à droite, l'IoU en fonction du seuil de décision.*

Le contour est défini comme la bande de **±2 pixels (±0,40 m) autour du périmètre de la vérité terrain**.

| Famille d'erreur | Part | Lecture |
|---|---|---|
| Faux positifs au contour | 31,1 % | débordement du masque prédit |
| Faux négatifs au contour | 25,1 % | retrait du masque prédit |
| **Sous-total contour** | **56,2 %** | **erreur de finesse géométrique** |
| Faux positifs loin | 28,1 % | détections là où il n'y a rien |
| Faux négatifs loin | 15,7 % | passages ou portions manqués |
| **Sous-total loin** | **43,8 %** | **erreur de détection** |

### Pourquoi c'est une bonne nouvelle pour le livrable SIG

Le livrable du projet n'est pas un masque : c'est **un point par passage piéton en Lambert-93**. Or une erreur qui se distribue le long du périmètre d'un objet déplace très peu son centroïde — un liséré de 2 px trop large sur un bord est en grande partie compensé par un liséré de 2 px trop étroit sur le bord opposé, et 2 px valent 40 cm au sol quand la cible d'acceptabilité est de **1,5 m**.

Autrement dit : **l'IoU de 0,638 est plus pessimiste que ce que vaut réellement le modèle pour la finalité métier**, puisqu'il pénalise au pixel près une classe d'erreur dont la vectorisation ponctuelle sera largement insensible.

> ⚠️ **Cet argument est une attente, pas une mesure.** L'erreur de centroïde n'a pas encore été calculée, et il reste 43,8 % d'erreurs *loin* du contour — celles-là, en revanche, créent ou suppriment des objets et frapperont de plein fouet le taux de détection comme la précision objet. Le raisonnement ne sera validé qu'une fois le chapitre de vectorisation écrit et l'erreur de centroïde mesurée en mètres.

---

## 🎚️ Un modèle calibré : le balayage de seuil est plat

| Seuil | 0,05 | 0,2 | 0,3 | 0,4 | **0,5** | **0,55** | 0,6 | 0,7 | 0,8 | 0,95 |
|---|---|---|---|---|---|---|---|---|---|---|
| IoU | 0,5787 | 0,6231 | 0,6323 | 0,6367 | **0,6384** | **0,6385** | 0,6383 | 0,6352 | 0,6275 | 0,5832 |

Sur toute la plage **0,35 – 0,65**, l'IoU reste à **moins de 0,004** de son optimum. L'optimum théorique (0,55, IoU 0,63850) ne bat le seuil par défaut (0,5, IoU 0,63837) que de **0,00013** — soit deux ordres de grandeur en dessous de la demi-largeur de l'intervalle de confiance (0,025).

Deux conséquences pratiques :

- **Le seuil reste à 0,5.** L'ajuster serait optimiser du bruit, et le faire sur le volet de validation puis publier le gain reviendrait à s'entraîner sur la validation.
- **Les probabilités du modèle sont bien séparées.** Une courbe plate autour de 0,5 signifie que peu de pixels vivent dans la zone d'incertitude ; le modèle tranche. C'est un résultat de robustesse : un seuil calé ici a des chances de rester valable ailleurs sur le territoire.

---

## 🔁 TTA D4 : +2 points d'IoU, mais pas sur la métrique qui compte

L'augmentation au test moyenne les probabilités prédites sur les **8 symétries du carré** (identité, 3 rotations, 4 réflexions), au prix de 8 inférences par tuile.

| | Sans TTA | Avec TTA D4 | Écart |
|---|---|---|---|
| IoU (seuil 0,5) | 0,6384 | **0,6579** | **+0,0195** (+3,1 %) |
| IoU au meilleur seuil | 0,6385 (à 0,55) | 0,6584 (à 0,45) | +0,0199 |
| Dice / F1 | 0,7793 | 0,7937 | +0,0144 |
| **Détection d'objet (> 50 %)** | **87,3 %** (165/189) | **85,7 %** (162/189) | **−1,6 point (−3 objets)** |
| Coût d'inférence | ×1 | **×8** | |

**Le gain n'est pas gratuit et il n'est pas uniforme.** Le moyennage des probabilités lisse les masques — d'où le gain sur les métriques de surface — mais ce même lissage érode les objets les plus faibles, dont trois passent sous le seuil de 50 % de couverture et cessent d'être comptés comme détectés. Sur la métrique cardinale du projet, la TTA **fait perdre**.

Arbitrage proposé, à revoir après le recollage en Lambert-93 :

- **En développement : jamais.** Multiplier par 8 le temps d'évaluation pour deux points d'une métrique qui n'est pas la métrique de décision n'a aucun intérêt.
- **En production : à trancher, pas à présumer.** Une passe unique sur le territoire peut absorber le coût, mais tant que la TTA dégrade le taux de détection, elle ne se justifie que si la vectorisation finale se révèle plus sensible à la qualité du contour qu'au nombre d'objets — ce qui reste à démontrer.

---

## 🎯 La métrique métier : 87,3 % de détection, avec une réserve

$$\text{Détection} = \frac{165}{189} = 87,3\ \% \qquad (\text{24 occurrences manquées})$$

Une occurrence est comptée détectée si **au moins 50 % de sa surface** est recouverte par la prédiction.

> ⚠️ **Ce pourcentage ne se lit pas « 87,3 % des passages piétons de Meudon ont été retrouvés ».** Le comptage se fait **par tuile**, et les tuiles se recouvrent à 50 % : un même passage piéton apparaît dans plusieurs tuiles et est compté plusieurs fois. Le nombre de passages **réels** distincts du volet est très inférieur aux 189 occurrences. Le comptage par objet de terrain — le seul qui intéresse la collectivité — exige le **recollage des prédictions en Lambert-93** via les fichiers `.tfw`, décidé au chapitre [`02/06`](../02_acquisition_de_donnees/06_decoupage_train_val_test.md) et **non encore réalisé**.

Une conséquence à garder en tête : le double comptage n'est pas neutre pour ce taux. Un passage bien vu compte plusieurs fois dans le numérateur, un passage systématiquement manqué compte plusieurs fois dans le dénominateur. Le chiffre après recollage peut aller dans les deux sens.

> 📌 **Point à vérifier avant publication** : les décomptes ne se recoupent pas d'une source à l'autre. Le chapitre [`02/06`](../02_acquisition_de_donnees/06_decoupage_train_val_test.md) annonce **200 occurrences pour 106 passages réels** sur `tiles_val`, `metrics_val.json` en compte **189**, et le diagnostic qualitatif annonce **59 passages réels**. Les trois chiffres sont produits par trois comptages différents (métadonnées ArcGIS, composantes connexes des masques, dédoublonnage géométrique) et l'écart n'a pas été instruit. **À élucider avant que le moindre taux de détection soit cité hors du dépôt.**

---

## 🖼️ Analyse qualitative

![Planche qualitative — meilleures, médianes et pires tuiles](../../datasets/runs/dinov3/figures/qualitative.png)
*Figure 3 : orthophoto, vérité terrain et prédiction pour les meilleures, médianes et pires tuiles. Vert = vrai positif, rouge = faux positif, bleu = faux négatif. Les 17 tuiles ne contenant qu'un fragment de bord ont été écartées de la planche.*

Quatre observations ressortent du diagnostic complet ([`diagnostic_val.html`](../../datasets/runs/dinov3/figures/diagnostic_val.html), les 88 tuiles triées) :

1. **Le régime nominal (IoU 0,70 – 0,83)** : les bandes sont retrouvées une à une, l'erreur se réduit à un liséré. Sur la meilleure tuile (IoU 0,826), 892 pixels attendus donnent 131 faux positifs et 47 faux négatifs — presque tous au contour.
2. **Les quatre échecs à IoU nul sont des artefacts de tuilage**, pas des échecs de modèle : trois tuiles ne contiennent que **22 à 74 pixels** de passage piéton, un fragment rogné par le bord de la tuile, que le modèle ignore complètement. Le constat rejoint celui de la [Phase 1](../03_phase_1_experience_pilote/01_duel_architectures_initial.md), où 100 % des échecs francs venaient déjà du bord de tuile. Le recollage en Lambert-93 fera disparaître ce mode d'échec : le passage tronqué ici est complet dans la tuile voisine.
3. **La quatrième tuile à IoU nul (`000000000489`) ne contient aucun passage piéton en vérité terrain** mais reçoit 156 pixels de faux positifs. C'est le seul aperçu, dans ce jeu, de ce que donnera le modèle sur une zone vide — et il est du mauvais côté. Rappel du chapitre [`02/06`](../02_acquisition_de_donnees/06_decoupage_train_val_test.md) : `ONLY_TILES_WITH_FEATURES` fait qu'**aucune tuile négative n'existe dans le jeu**. La précision objet mesurée ici est donc structurellement optimiste.
4. **Les pires tuiles non triviales (IoU 0,28 – 0,29) décrivent le même carrefour**, où le modèle pose des détections franches sur des marquages au sol qui n'en sont pas. C'est le mode d'échec le plus coûteux pour le livrable SIG : un faux positif compact et confiant deviendra un point dans la couche finale.

---

## ⚠️ Limites assumées

1. **Une seule graine.** Aucune mesure de dispersion. Le plateau des époques 11-29 oscille de ±0,02 d'IoU d'une époque à l'autre ; rien ne dit pour l'instant que l'écart entre deux architectures dépassera cette amplitude. Trois graines sont prévues (`--seed 0 1 2`).
2. **La validation repose sur une seule commune.** Meudon-sur-Seine est une bande étroite en bord de Seine, au bâti et à la voirie particuliers. Le chiffre répond à « le modèle transfère-t-il à Meudon-sur-Seine », pas « à une commune quelconque ».
3. **Aucune tuile négative dans le jeu.** Le modèle n'a jamais vu de rue vide, de parking ou de toiture sans passage piéton, ni à l'entraînement ni à l'évaluation. Point ouvert hérité du chapitre [`02/06`](../02_acquisition_de_donnees/06_decoupage_train_val_test.md), à trancher avant l'inférence sur le territoire complet.
4. **Comptage par occurrence, pas par objet de terrain**, et décomptes divergents entre sources (voir plus haut).
5. **Les métriques cardinales du projet n'ont pas de chiffre** : ni erreur de centroïde, ni MAE de comptage, ni précision objet. Seul le rappel objet est mesuré.
6. **Le volet test n'a pas été ouvert** — ce qui est le comportement voulu, mais implique qu'**aucun chiffre de généralisation n'existe encore** dans ce projet.
7. **La comparaison avec le pilote n'est pas stricte** : jeux de validation différents (117 tuiles / 312 occurrences contre 88 / 189), architecture U-Net du pilote entraînée par ArcGIS sur un découpage aléatoire inconnu. Les écarts s'interprètent en tendance, pas au centième.

---

## 📌 Ce que ce run autorise à dire

✅ **Un DINOv3 gelé, dont 0,8 % des paramètres seulement sont entraînés, atteint 0,638 d'IoU et 87,3 % de détection d'occurrence sur une commune jamais vue**, en 4 heures sur un portable Apple Silicon, sans une seule ligne de fine-tuning du backbone.

✅ **Ce chiffre est le premier du projet à être exempt de fuite spatiale.** Il est plus bas que celui du pilote parce qu'il est plus vrai.

✅ **Le modèle est calibré** (seuil sans effet entre 0,35 et 0,65) et **son erreur est majoritairement géométrique** (56 % au contour), deux propriétés favorables à un livrable ponctuel.

❌ **Ce run ne dit rien sur la comparaison des architectures** : le U-Net ResNet34 n'a pas encore tourné sous ce protocole.

❌ **Ce run ne dit rien sur la performance finale du projet** : le test est fermé, les métriques ponctuelles ne sont pas calculées, la couche SIG n'existe pas.
