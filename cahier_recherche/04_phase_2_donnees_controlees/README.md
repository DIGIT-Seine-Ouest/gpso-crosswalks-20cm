# Phase 2 : Protocole expérimental à données strictement contrôlées

> **Statut** : Protocole figé, prêt pour exécution.  
> **Rôle** : Réaliser l'expérience scientifique définitive sans aucune fuite spatiale, avec mesure de la variabilité statistique (multi-seed) et production du référentiel SIG ponctuel final.

---

## 🛡️ Les 3 piliers de la Phase 2

```mermaid
flowchart TD
    subgraph P2 ["Cadre d'Expérimentation Contrôlée (Phase 2)"]
        direction TB
        PIL1["1. Split Spatial Étanche (0,0 % fuite)<br>393 Train (Ouest) | 88 Val (Meudon) | 101 Test (Issy/Vanves)"]
        PIL2["2. Entraînement Multi-Seed (3 runs)<br>Seeds 0, 1, 2 pour DINOv3 et U-Net<br>Publication en Moyenne ± Écart-type"]
        PIL3["3. Vectorisation Ponctuelle SIG<br>Du masque aux points Lambert-93 (EPSG:2154)<br>Mesure de l'erreur submétrique de centroïde"]
    end

    style P2 fill:#f1f8e9,stroke:#558b2f,stroke-width:2px
    style PIL1 fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px
    style PIL2 fill:#e3f2fd,stroke:#1565c0,stroke-width:1px
    style PIL3 fill:#fff3e0,stroke:#e65100,stroke-width:1px
```

---

## 📚 Documents de la Phase 2

- [`01_protocole_spatial_disjoint.md`](01_protocole_spatial_disjoint.md) : Découpage spatial v2, scripts d'entraînement et règles d'étanchéité du test à l'aveugle.
- [`02_pipeline_vectorisation_sig.md`](02_pipeline_vectorisation_sig.md) : Algorithme de vectorisation ponctuelle, calcul des centroïdes en Lambert-93 et schéma de données SIG.
