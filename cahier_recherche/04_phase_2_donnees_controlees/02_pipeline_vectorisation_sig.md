# 02 — Chaîne de vectorisation ponctuelle : du masque binaire aux points SIG

## 🎯 Objectif final
Convertir la carte de probabilités générée par le réseau de Deep Learning en une **couche vectorielle de points géoréférencés (coordonnées Lambert-93 EPSG:2154)** directement exploitable par les services techniques et SIG de GPSO.

---

## ⚙️ Les 5 étapes de l'algorithme de vectorisation

```mermaid
flowchart TD
    M["1. Masque de Probabilités IA<br>(Matrice 512×512 en flottants [0, 1])"] --> BIN["2. Seuillage & Binarisation<br>(Seuil optimal τ = 0,50)"]
    BIN --> MORPH["3. Nettoyage Morphologique & Filtrage<br>(Fermeture + suppression îlots < 100 px / 4 m²)"]
    MORPH --> COMP["4. Extraction Composantes Connexes<br>& Calcul Centroïde Pixel (X_px, Y_px)"]
    COMP --> GEO["5. Projection Cartographique .tfw<br>Coordonnées Métriques Lambert-93 (EPSG:2154)"]
    GEO --> OUT["6. Couche Vectorielle Ponctuelle SIG<br>(passages_pietons_points.geojson)"]

    style M fill:#ede7f6,stroke:#512da8,stroke-width:2px
    style BIN fill:#e3f2fd,stroke:#1565c0,stroke-width:1px
    style MORPH fill:#fff3e0,stroke:#e65100,stroke-width:1px
    style COMP fill:#fff8e1,stroke:#f57f17,stroke-width:1px
    style GEO fill:#e1f5fe,stroke:#0288d1,stroke-width:1px
    style OUT fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

---

## 1. Binarisation & Seuillage
Le réseau produit une matrice de probabilités continues $[0.0, 1.0]$. Un seuil $\tau = 0,50$ sépare le fond des rayures positives.

---

## 2. Filtrage morphologique
- **Fermeture morphologique (Closing)** : comble les éventuels trous fins entre les rayures d'un même passage.
- **Filtrage par aire minimale** : Élimination des composantes connexes $< 100\text{ pixels}$ ($4\text{ m}^2$ au sol), trop petites pour être des passages piétons réglementaires.

---

## 3. Calcul du centroïde et géoréférencement en Lambert-93
Pour chaque composante connexe $k$ de $N_k$ pixels :
$$X_{c, \text{pixel}} = \frac{1}{N_k} \sum_{i=1}^{N_k} x_i, \quad Y_{c, \text{pixel}} = \frac{1}{N_k} \sum_{i=1}^{N_k} y_i$$

Puis conversion en mètres Lambert-93 (EPSG:2154) via le fichier de calage `.tfw` :
$$\begin{pmatrix} X_{\text{Lambert93}} \\ Y_{\text{Lambert93}} \end{pmatrix} = \begin{pmatrix} X_0 \\ Y_0 \end{pmatrix} + \begin{pmatrix} +0.20 & 0 \\ 0 & -0.20 \end{pmatrix} \begin{pmatrix} X_{c, \text{pixel}} \\ Y_{c, \text{pixel}} \end{pmatrix}$$

---

## 📋 Schéma d'attributs de la couche SIG exportée

| Champ | Type | Description | Exemple |
|---|---|---|---|
| `id_passage` | String | Identifiant unique national/territorial | `GPSO_CW_00482` |
| `coord_x_l93` | Float | Coordonnée X Lambert-93 (mètres) | `644527.42` |
| `coord_y_l93` | Float | Coordonnée Y Lambert-93 (mètres) | `6857652.18` |
| `aire_sol_m2` | Float | Surface estimée du passage piéton | `18.4 m²` |
| `confiance_ia` | Float | Score moyen de probabilité DINOv3 | `0.94` |
| `commune_estimee` | String | Commune de rattachement | `Meudon` |
| `date_detection` | Date | Date du run d'inférence | `2026-08-20` |
