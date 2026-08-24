"""Chemins, constantes geometriques et normalisations.

Tout ce qui est susceptible de changer d'une machine a l'autre est ici, et
surchargeable par variable d'environnement : aucun autre module ne construit
de chemin en dur.
"""
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Racine des jeux de donnees. Contient tiles_train/, tiles_val/, tiles_test/,
# produits par le chapitre 06 du cahier de recherche.
DATA_ROOT = os.environ.get("GPSO_DATA", os.path.join(REPO, "datasets"))

# Nom du dossier par volet. Le decoupage est spatial et fige a l'export : il n'y
# a plus de fichier de split a charger, chaque volet est un dossier a part.
SUBSET_DIRS = {
    "train": "tiles_train",
    "val": "tiles_val",
    "test": "tiles_test",
}
SUBSETS = tuple(SUBSET_DIRS)

# Jeu du pilote : un seul dossier, decoupe apres coup par un split.json.
# Conserve pour pouvoir rejouer la Phase 1, jamais pour produire un resultat.
LEGACY_TILES = os.environ.get("GPSO_TILES", os.path.join(DATA_ROOT, "tiles"))

# Geometrie des tuiles, imposee par l'export ArcGIS (cf. cahier 02/06).
IMG_SIZE = 512
PIXEL_SIZE_M = 0.20
TILE_SIZE_M = IMG_SIZE * PIXEL_SIZE_M       # 102.4 m de cote
EPSG = 2154                                  # Lambert-93

# Normalisations. Chaque modele recoit le pretraitement de son pre-entrainement :
# ce n'est pas une variable libre du protocole de comparaison.
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
SAT493M = ((0.430, 0.411, 0.296), (0.213, 0.156, 0.143))
NO_NORM = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

NORMALIZATIONS = {"imagenet": IMAGENET, "sat493m": SAT493M, "none": NO_NORM}


def subset_dir(subset, data_root=None):
    """Chemin du dossier d'un volet. Erreur explicite si le volet est inconnu."""
    if subset not in SUBSET_DIRS:
        raise ValueError(f"volet inconnu : {subset!r} (attendus : {', '.join(SUBSETS)})")
    return os.path.join(data_root or DATA_ROOT, SUBSET_DIRS[subset])


# Sorties d'entrainement : checkpoints, historiques, rapports d'evaluation, et
# store MLflow. Elles vivent sous DATA_ROOT, qui est le miroir local du bucket
# Hugging Face : un seul dossier synchronise porte les entrees et les sorties,
# et passer du Mac a Colab ne demande qu'un `python -m scripts.hf_sync pull`.
RUNS_DIR = os.environ.get("GPSO_RUNS", os.path.join(DATA_ROOT, "runs"))
MLRUNS_DIR = os.environ.get("GPSO_MLRUNS", os.path.join(DATA_ROOT, "mlruns"))

#: bucket de reference, cible par defaut de scripts/hf_sync.py
BUCKET = os.environ.get("GPSO_BUCKET", "mandresyandri/gpso-crosswalks-20cm")
