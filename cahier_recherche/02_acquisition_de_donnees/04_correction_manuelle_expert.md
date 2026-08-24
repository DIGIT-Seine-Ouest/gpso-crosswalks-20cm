# 04 — Protocole de correction manuelle & nettoyage expert sous SIG

Pour obtenir une vérité terrain (*Ground Truth*) irréprochable, l'ensemble des propositions de SAM 3 a été inspecté et corrigé visuellement sous SIG par un opérateur humain.

---

## 1. Méthode et règles d'édition topologique

Chaque proposition brute a été soumise à un protocole systématique :

```mermaid
flowchart TD
    P["Proposition Brute SAM 3"] --> DEC{Nature de l'anomalie}
    
    DEC -->|Faux positif : flèche, îlot, artefact| SUPP["Action : Suppression définitive<br>(Outil Delete)"]
    DEC -->|Deux passages collés ou fusionnés| DIV["Action : Séparation en 2 entités<br>(Outil Divide / Split)"]
    DEC -->|Passage tronqué par une ombre| MERGE["Action : Assemblage continu<br>(Outil Merge / Union)"]
    DEC -->|Bordure décalée ou imprécise| REC["Action : Ajustement des sommets<br>(Snapping sur les rayures)"]

    SUPP --> OUT["Base Validée : 672 passages piétons nets"]
    DIV --> OUT
    MERGE --> OUT
    REC --> OUT

    style P fill:#ede7f6,stroke:#512da8,stroke-width:2px
    style SUPP fill:#ffebee,stroke:#c62828,stroke-width:1px
    style DIV fill:#fff3e0,stroke:#e65100,stroke-width:1px
    style MERGE fill:#e3f2fd,stroke:#1565c0,stroke-width:1px
    style REC fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px
    style OUT fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

1. **Suppression des faux positifs (`Delete`)** :
   - Élimination des flèches directionnelles, zébras de stationnement, îlots directionnels et marquages provisoires de travaux.
2. **Division des passages fusionnés (`Divide / Split Tool`)** :
   - Quand deux passages piétons distincts situés à moins de 2 mètres ont été englobés dans un unique polygone par SAM, ils ont été scindés en **deux entités distinctes**.

![Exemple de fusion par SAM nécessitant l'outil Divide](../../assets/sam_fusion_carrefour_besoin_divide.png)
*Figure 1 : Deux passages piétons distincts au carrefour fusionnés en un seul polygone (OID 70) — corrigé avec l'outil Divide.*

![Fusion complexe de passages piétons et piste cyclable](../../assets/sam_fusion_complexe_croisement.png)
*Figure 2 : Cas complexe où SAM a fusionné 3 traversées piétonnes et cyclables en une entité géante de 83,5 m².*

![Passage fusionné avec un chevron de rampe](../../assets/sam_fusion_passage_et_chevron_rampe.png)
*Figure 3 : Passage piéton fusionné avec un chevron autoroutier (OID 298 de 228 m²) — scindé par Divide puis suppression du chevron.*

3. **Fusion des fragments (`Merge / Union Tool`)** :
   - Quand l'ombre d'un candélabre ou d'un immeuble a sectionné un passage piéton en deux sous-polygones, les entités ont été fusionnées en **un polygone continu**.

![Vue large rue canyon et ombre d'immeuble](../../assets/sam_ombre_immeuble_vue_large.png)
*Figure 4 : Rue canyon où l'ombre portée d'un bâtiment coupe un passage piéton en deux.*

![Zoom sur le passage coupé par l'ombre](../../assets/sam_ombre_immeuble_zoom_besoin_merge.png)
*Figure 5 : Zoom sur les deux polygones disjoints (OID 104 & 108) — assemblés en une seule entité par l'outil Merge.*

![Passage fragmenté par l'ombre d'un arbre](../../assets/sam_fragmentation_ombre.png)
*Figure 6 : Autre cas de fragmentation sous le feuillage d'un arbre.*

4. **Recalage fin des contours & Audit des attributs** :
   - Ajustement des sommets pour caler le polygone sur l'empreinte exacte des bandes blanches peintes sur la chaussée.

![Passage piéton avec îlot refuge central](../../assets/sam_passage_ilot_refuge.png)
*Figure 7 : Découpage propre d'un passage piéton traversant scindé par un îlot refuge central.*

![Audit de la table d'attributs sous ArcGIS Pro](../../assets/sam_attributs_carrefour.png)
*Figure 8 : Contrôle systématique des polygones et des surfaces dans la table attributaire.*

---

## 2. Décompte quantitatif des annotations validées

Sur le premier quartier pilote, la correction humaine a fait passer le jeu d'entités de **244 propositions brutes à 215 passages piétons vérifiés**.

La campagne d'annotation sur l'ensemble des quartiers du territoire a produit la série suivante :

| Quartier / Secteur | Entités validées | Description |
|---|---|---|
| Quartier 1 (Pilote) | **215** | Secteur urbain dense (Boulogne Nord / Centre) |
| Quartier 2 | **178** | Secteur mixte commercial et résidentiel (Boulogne Sud) |
| Quartier 3 | **125** | Secteur Est (Issy-les-Moulineaux / Vanves) |
| Quartier 4 | **107** | Secteur Sud (Meudon centre / plateau) |
| Quartier 5 | **36** | Secteur Ouest (Sèvres / Chaville) |
| Quartier 6 | **11** | Secteur boisé et limites périurbaines |
| **TOTAL GÉNÉRAL** | **672 passages piétons** | **Couche vectorielle `passages_pietons_all`** |

---

## 3. Règle de classification
Toutes les entités validées ont été affectées à une classe unique :
- **Nom de classe** : `Undefined` (ou `pedestrian_crossing`).
- **Valeur numérique raster** : `1` (positive).
- **Valeur de fond** : `0` (négative / NoData / chaussée).
