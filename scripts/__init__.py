"""Detection de passages pietons sur orthophotographie 20 cm (territoire GPSO).

Le paquet est organise pour qu'aucun module transverse ne connaisse un modele
en particulier :

    config.py    chemins, normalisations, constantes geometriques
    dataset.py   lecture des tuiles et des masques, augmentation D4
    metrics.py   metriques pixel et objet, TTA, rapport
    models/      un module par architecture, declares dans un registre

Les points d'entree (train.py, evaluate.py, visualize.py) ne manipulent que
l'interface commune definie par models.base.SegModel. Ajouter une architecture
ne demande donc de toucher a aucun des trois.
"""

__version__ = "2.0.0"
