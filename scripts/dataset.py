"""Lecture des tuiles et des masques, identique pour tous les modeles.

Un seul code lit les donnees : si deux modeles obtiennent des scores differents,
l'ecart vient des modeles et pas de la facon dont on leur a servi les images.

Le decoupage train/val/test est spatial et fige a l'export ArcGIS (un dossier par
volet, cf. cahier de recherche 02/06). Il n'y a donc rien a decouper ici : la
fonction load_stems se contente de lister le dossier demande.
"""
import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import config


def load_stems(subset, data_root=None):
    """Identifiants des tuiles d'un volet, apparies image <-> masque.

    Une tuile sans masque (ou l'inverse) est ecartee silencieusement : c'est le
    seul cas ou un fichier orphelin ne doit pas faire echouer un entrainement.
    """
    root = config.subset_dir(subset, data_root)
    if not os.path.isdir(root):
        raise SystemExit(
            f"Volet {subset!r} introuvable : {root}\n"
            f"Attendu : un dossier {config.SUBSET_DIRS[subset]}/ contenant images/ et labels/.\n"
            f"Surcharge la racine avec GPSO_DATA=/chemin/vers/datasets si besoin."
        )
    imgs = {_stem(p) for p in glob.glob(os.path.join(root, "images", "*.tif"))}
    lbls = {_stem(p) for p in glob.glob(os.path.join(root, "labels", "*.tif"))}
    stems = sorted(imgs & lbls)
    if not stems:
        raise SystemExit(f"Aucune paire image/masque sous {root}")
    orphans = len(imgs ^ lbls)
    if orphans:
        print(f"[data] {orphans} fichier(s) sans contrepartie ignore(s) dans {subset}")
    return stems


def load_stems_from_split(split_json, subset):
    """Mode heritage : un seul dossier de tuiles, decoupe par un fichier JSON.

    Ne sert qu'a rejouer l'experience pilote de la Phase 1, dont le decoupage
    etait aleatoire et fuyait. Ne pas utiliser pour produire un resultat.
    """
    import json

    with open(split_json) as f:
        split = json.load(f)
    if not isinstance(split.get(subset), list):
        dispo = [k for k, v in split.items() if isinstance(v, list)]
        raise SystemExit(f"'{subset}' absent de {os.path.basename(split_json)} ; "
                         f"disponibles : {dispo}")
    return split[subset]


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def fingerprint(subset, data_root=None, split_json=None):
    """Empreinte de la composition d'un volet : noms des tuiles et tailles des
    fichiers, condensees en SHA-256.

    Deux entrainements qui affichent la meme empreinte ont vu exactement les
    memes octets. Un export refait, une tuile ecrasee par une homonyme, un
    fichier tronque a mi-copie : l'empreinte change, et deux runs cessent d'etre
    comparables sans qu'on ait a s'en apercevoir apres coup.

    C'est peu couteux — une lecture de taille par fichier, pas de contenu — et
    c'est ce que le tracage enregistre en parametre de chaque run.
    """
    import hashlib

    if split_json:
        stems = sorted(load_stems_from_split(split_json, subset))
        root = config.LEGACY_TILES
    else:
        stems = load_stems(subset, data_root)
        root = config.subset_dir(subset, data_root)

    h = hashlib.sha256()
    for stem in stems:
        tailles = []
        for kind in ("images", "labels"):
            p = os.path.join(root, kind, f"{stem}.tif")
            tailles.append(os.path.getsize(p) if os.path.exists(p) else -1)
        h.update(f"{stem}:{tailles[0]}:{tailles[1]}\n".encode())
    return h.hexdigest()


class TileSet(Dataset):
    """Paires (image normalisee, masque entier) d'un volet.

    L'augmentation D4 (les 8 symetries du carre) est la seule transformation
    appliquee, et elle l'est a l'identique pour tous les modeles. La
    normalisation, elle, est propre au modele : elle est passee en argument.

    Les masques sont ecrits par ArcGIS en entiers non signes sur 16 bits, alors
    que les images sont en 8 bits sur trois bandes. Les valeurs utiles restent
    {0, 1} ; un lecteur qui suppose du 8 bits partout obtient un masque etire et
    decale sans lever la moindre exception, d'ou la verification ci-dessous.
    """

    def __init__(self, stems, mean, std, augment=False, root=None, subset=None,
                 data_root=None, check_masks=True):
        if root is None:
            if subset is None:
                raise ValueError("TileSet demande soit root=, soit subset=")
            root = config.subset_dir(subset, data_root)
        self.stems = list(stems)
        self.root = root
        self.augment = augment
        self.mean = np.asarray(mean, np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, np.float32).reshape(3, 1, 1)
        if check_masks and self.stems:
            self._check_mask(self.stems[0])

    def _check_mask(self, stem):
        msk = np.array(Image.open(self._path("labels", stem)))
        vals = np.unique(msk)
        if not set(vals.tolist()) <= {0, 1}:
            raise SystemExit(
                f"Masque {stem} : valeurs {vals.tolist()[:8]} au lieu de {{0, 1}}.\n"
                f"Un masque ArcGIS Classified_Tiles est en uint16 de valeurs 0/1 ; "
                f"des valeurs aberrantes signalent une lecture en 8 bits ou un export "
                f"dans un autre format de metadonnees."
            )

    def _path(self, kind, stem):
        return os.path.join(self.root, kind, f"{stem}.tif")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.stems[i]
        img = np.array(Image.open(self._path("images", stem)).convert("RGB"))   # HWC uint8
        msk = np.array(Image.open(self._path("labels", stem)))                  # HW uint16
        if self.augment:
            img, msk = _augment_d4(img, msk)
        img = np.ascontiguousarray(img.transpose(2, 0, 1), np.float32) / 255.0
        img = (img - self.mean) / self.std
        msk = np.ascontiguousarray(msk).astype(np.int64)   # uint16 -> int64 pour la CE
        return torch.from_numpy(img), torch.from_numpy(msk)


def _augment_d4(img, msk):
    """Flips et rotations de 90 degres : le groupe des symetries du carre.

    Une orthophoto n'a pas de haut ni de bas, les 8 orientations sont donc toutes
    plausibles. C'est la seule augmentation du protocole, et elle est commune a
    tous les modeles.
    """
    if np.random.rand() < 0.5:
        img, msk = img[:, ::-1], msk[:, ::-1]
    if np.random.rand() < 0.5:
        img, msk = img[::-1], msk[::-1]
    k = np.random.randint(4)
    if k:
        img, msk = np.rot90(img, k, (0, 1)), np.rot90(msk, k, (0, 1))
    return img, msk


def make_loader(subset, mean, std, batch_size=2, augment=None, shuffle=None,
                workers=0, data_root=None, split_json=None, drop_last=False):
    """DataLoader d'un volet. Par defaut, seul le train est augmente et melange."""
    if split_json:
        stems = load_stems_from_split(split_json, subset)
        root = config.LEGACY_TILES
    else:
        stems = load_stems(subset, data_root)
        root = config.subset_dir(subset, data_root)
    is_train = subset == "train"
    ds = TileSet(stems, mean, std,
                 augment=is_train if augment is None else augment,
                 root=root)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=is_train if shuffle is None else shuffle,
                      num_workers=workers, drop_last=drop_last and is_train)


def read_tfw(path):
    """Coordonnees de calage d'une tuile : (taille du pixel, X, Y du coin haut-gauche).

    Sert a repositionner une prediction dans le referentiel Lambert-93, seule
    facon de compter un passage pieton une fois et une seule quand il apparait
    sur plusieurs tuiles chevauchantes.
    """
    with open(path) as f:
        v = [float(x) for x in f.read().split()]
    return {"px": v[0], "py": v[3], "x0": v[4], "y0": v[5]}


def tile_extent(stem, subset=None, root=None, data_root=None):
    """Emprise (x_min, y_min, x_max, y_max) d'une tuile, en metres Lambert-93."""
    root = root or config.subset_dir(subset, data_root)
    t = read_tfw(os.path.join(root, "images", f"{stem}.tfw"))
    return (t["x0"], t["y0"] + config.IMG_SIZE * t["py"],
            t["x0"] + config.IMG_SIZE * t["px"], t["y0"])
