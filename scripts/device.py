"""Choix du peripherique de calcul et reglages de precision associes.

Un seul module decide ou tourne un entrainement et dans quelle precision.
train.py, evaluate.py et visualize.py s'en remettent a lui ; aucun module de
modele n'a a connaitre le peripherique.

Trois cibles :

  cuda  la reference du protocole (T4 Colab, RTX Ada). Precision mixte fp16
        avec GradScaler, activee par defaut.
  mps   les GPU Apple Silicon, en memoire unifiee. Pas de GradScaler : il est
        specifique a CUDA, et l'echelle de gradient n'a pas lieu d'etre quand on
        entraine en fp32. L'autocast fp16 y reste facultatif (--amp), certains
        noyaux etant encore instables selon la version de torch.
  cpu   dernier recours, hors de question pour un ViT-L.

Ce que le peripherique NE change PAS : les tuiles, l'augmentation, la loss, le
nombre d'epoques, la selection du checkpoint, les metriques. Un IoU obtenu sur
MPS et un IoU obtenu sur T4 restent donc comparables, aux arrondis pres.

Ce qu'il change : le temps par epoque. La telemetrie d'un run MPS ne se compare
pas a celle d'un run CUDA — d'ou l'enregistrement du peripherique dans
l'historique et dans le checkpoint.
"""
import contextlib

import torch

CHOIX = ("cuda", "mps", "cpu")


def disponibles():
    """Peripheriques utilisables sur cette machine, du plus rapide au moins."""
    dispo = []
    if torch.cuda.is_available():
        dispo.append("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dispo.append("mps")
    dispo.append("cpu")
    return dispo


def pick(demande=None):
    """Peripherique a utiliser. Sans demande explicite, le plus rapide presente.

    Une demande explicite indisponible est une erreur, pas un repli silencieux :
    decouvrir apres deux heures qu'un entrainement a tourne en CPU parce que
    --device cuda avait ete ignore est le genre de surprise que ce depot evite.
    """
    dispo = disponibles()
    if demande is None:
        return dispo[0]
    if demande not in CHOIX:
        raise SystemExit(f"Peripherique inconnu : {demande!r} "
                         f"(attendus : {', '.join(CHOIX)})")
    if demande not in dispo:
        raise SystemExit(f"Peripherique {demande!r} indisponible sur cette machine "
                         f"(disponibles : {', '.join(dispo)})")
    return demande


def use_amp(device, no_amp=False, amp=False):
    """Faut-il activer la precision mixte fp16 ?

    Par defaut : oui sur cuda, non ailleurs. --no-amp la coupe sur cuda, --amp la
    force sur mps pour qui veut tenter le gain de vitesse.
    """
    if no_amp:
        return False
    if device == "cuda":
        return True
    return bool(amp) and device == "mps"


def autocast(device, enabled):
    """Contexte de precision mixte, ou un contexte vide si elle est desactivee."""
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device, dtype=torch.float16)


def grad_scaler(device, enabled):
    """GradScaler la ou il existe.

    Il est propre a CUDA. Sur mps et cpu on renvoie un scaler desactive, qui se
    comporte en simple passe-plat : la boucle d'entrainement reste unique.
    """
    return torch.amp.GradScaler("cuda", enabled=enabled and device == "cuda")


def describe(device, amp):
    """Ligne d'information affichee au lancement."""
    if device == "cuda":
        nom = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        detail = f"{nom}, {vram:.1f} Go"
    elif device == "mps":
        detail = "Apple Silicon, memoire unifiee"
    else:
        detail = "processeur"
    return f"{device} ({detail}) | precision mixte fp16 : {'oui' if amp else 'non'}"
