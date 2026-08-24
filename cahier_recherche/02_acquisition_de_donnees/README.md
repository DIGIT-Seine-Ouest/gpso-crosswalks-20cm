# Chaîne d'acquisition et préparation des données : de `raw_imgs/` à `tiles/`

Ce dossier documente de manière exhaustive, reproductible et définitive l'ensemble de la chaîne de traitement géospatial et d'annotation ayant permis de transformer les 7 dalles brutes d'orthophotographies aériennes en un jeu de données de 582 vignettes d'apprentissage pour le Deep Learning.

---

## 🗺️ Schéma global du flux de données

```mermaid
flowchart TD
    A["7 Dalles BD ORTHO 20 cm<br>(raw_imgs/)"] --> M["Mosaic To New Raster<br>(Lambert-93, 0.20 m/px)"]
    
    subgraph S_EMPRISE ["Préparation de l'emprise territoriale"]
        E["Emprise GPSO"] --> D["Dissolve"] --> B["Buffer 50m"]
    end
    
    B --> M
    M --> C["Clip Raster<br>(Ortho Mosaïquée Découpée)"]
    C --> SAM["Pré-annotation SAM 3<br>(Zero-shot : prompt 'pedestrian crossing')"]
    SAM --> EDIT["Nettoyage & Correction Manuelle SIG<br>(Divide, Merge, Filtrage faux positifs)<br>👉 672 passages validés"]
    EDIT --> EXP["Export Training Data For DL<br>(512×512 px, Stride 256 px, Classified_Tiles)"]
    EXP --> TILES["Dataset Final (tiles/)<br>582 paires Images / Masques"]

    style A fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style SAM fill:#ede7f6,stroke:#512da8,stroke-width:2px
    style EDIT fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style TILES fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

---

## 📚 Guide des chapitres

| Chapitre | Titre | Description |
|---|---|---|
| [`01_selection_ortho_ign.md`](01_selection_ortho_ign.md) | **Sélection & Caractéristiques des Orthophotos IGN** | Les 7 dalles BD ORTHO 2024, système Lambert-93, justification physique de la résolution 20 cm. |
| [`02_traitements_sig_arcgis.md`](02_traitements_sig_arcgis.md) | **Traitements Géospatiaux sous ArcGIS Pro** | Dissolve, Buffer 50 m, Mosaic To New Raster et Clip Raster. |
| [`03_pre_annotation_sam3.md`](03_pre_annotation_sam3.md) | **Pré-annotation assistée par IA (SAM 3)** | Utilisation du modèle de fondation SAM 3 zero-shot et analyse des premières détections. |
| [`04_correction_manuelle_expert.md`](04_correction_manuelle_expert.md) | **Protocole de Correction & Nettoyage SIG** | Opérations topologiques (`Divide`, `Merge`), suppression des faux positifs, décompte des 672 objets. |
| [`05_tuilage_export_tiles.md`](05_tuilage_export_tiles.md) | **Génération des Vignettes & Structure `tiles/`** | Export en tuiles 512×512, pas de 256 px, formats raster et métadonnées géoréférencées. |

---

## 📦 Résumé des données produites dans [`tiles/`](../../tiles/)

- **582 vignettes images** (`tiles/images/*.tif` + `.tfw`) : RVB 8-bit non signé, $512 \times 512$ pixels.
- **582 masques de segmentation** (`tiles/labels/*.tif` + `.tfw`) : Valeurs {0 = Fond, 1 = Passage piéton}.
- **Résolution spatiale** : $0,20\text{ m}$ par pixel ($102,4\text{ m} \times 102,4\text{ m}$ par vignette).
- **Proportion de pixels positifs** : $0,6725\ \% \approx 1\ 026\ 039\text{ pixels}$.
