# 01 — Sélection et caractéristiques des orthophotographies IGN

## 1. Origine et sélection des dalles

Les images sources proviennent de la **BD ORTHO®** de l'IGN (Institut national de l'information géographique et forestière), millésime 2024.

Pour couvrir l'ensemble du territoire de l'établissement public territorial **Grand Paris Seine Ouest (GPSO)** (Boulogne-Billancourt, Issy-les-Moulineaux, Meudon, Vanves, Sèvres, Chaville, Ville-d'Avray, Marnes-la-Coquette), **7 dalles géoréférencées** ont été sélectionnées.

```powershell
# Script d'archivage des dalles brutes dans raw_imgs/
New-Item -ItemType Directory -Force -Path "raw_imgs"

$files = @(
  "92-2024-0645-6865-LA93-0M20-E080.jp2",
  "92-2024-0645-6860-LA93-0M20-E080.jp2",
  "92-2024-0640-6865-LA93-0M20-E080.jp2",
  "92-2024-0640-6860-LA93-0M20-E080.jp2",
  "92-2024-0640-6855-LA93-0M20-E080.jp2",
  "92-2024-0635-6865-LA93-0M20-E080.jp2",
  "92-2024-0635-6860-LA93-0M20-E080.jp2"
)

Copy-Item -Path $files -Destination "raw_imgs/"
```

- **Format brut** : JPEG2000 (`.jp2`) avec compression avec pertes sans dégradation visuelle.
- **Poids total** : ~631 Mo.
- **Système de coordonnées (SCR)** : RGF93 / Lambert-93 (code EPSG : **2154**).

---

## 2. Pourquoi la résolution 20 cm ($0,20\text{ m/pixel}$) est critique ?

La nomenclature du fichier contient `0M20`, indiquant une taille de pixel au sol de **$20\text{ cm} \times 20\text{ cm}$**.

### Justification géométrique :
En milieu urbain en France, les bandes blanches transversales d'un passage piéton réglementaire (zébras) ont une largeur normalisée de **$40\text{ à }50\text{ cm}$** et un espacement de $50\text{ cm}$ :

| Résolution | Pixels par bande blanche | Visibilité du passage piéton |
|---|---|---|
| **$0,20\text{ m}$ (BD ORTHO retenue)** | **2 à 3 pixels** | **Net et contrasté** : L'alternance blanc/noir du motif zébré est nettement résolue par le modèle. |
| **$0,50\text{ m}$ (Imagerie standard)** | ~1 pixel | **Flou / Dégradé** : Le blanc et le bitume se mélangent dans les mêmes pixels (effet de mélange spectral). |
| **$1,00\text{ m}$ (Satellite basse rés.)** | < 0,5 pixel | **Invisibilité totale** : Le passage piéton disparaît complètement dans la texture de la chaussée. |

> 📌 **Conclusion** : La résolution de $0,20\text{ m}$ est le seuil physique minimal permettant à un réseau de neurones de reconnaître la morphologie rayée caractéristique d'un passage piéton.
