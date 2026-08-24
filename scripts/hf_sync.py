"""Synchronisation du dossier datasets/ avec le bucket Hugging Face.

    python -m scripts.hf_sync pull            # bucket -> local, avant d'entrainer
    python -m scripts.hf_sync push            # local -> bucket, apres
    python -m scripts.hf_sync push --dry-run  # ce qui partirait, sans rien envoyer

Un seul bucket porte tout : les entrees (raw_imgs/, tiles_*/) et les sorties
(runs/, mlruns/). Passer du Mac a Colab tient donc dans un `pull` au debut et un
`push` a la fin, sans rien recopier a la main.

Ce module enveloppe `hf buckets sync`, qui fait deja le travail : comparaison
des deux cotes, transfert des seuls fichiers changes, filtres, plan revisable.
Il n'ajoute que ce que la ligne de commande nue oublierait.

L'option qui n'est pas negociable
---------------------------------
`--ignore-times`. Le bucket ne renvoie pas de date de modification : il repond
`1970-01-01` pour chaque objet. La comparaison par defaut croise taille ET date,
et classe donc les 2 837 fichiers en « local plus recent » — un `push` nu
reenvoie tout le jeu a chaque fois. En comparant les tailles seules, seuls les
fichiers reellement modifies partent. Ce module pose toujours le drapeau.

Ce qui n'est jamais envoye
--------------------------
Les fichiers d'exploitation du systeme, et les checkpoints si `--no-weights` est
passe : un `.pt` de U-Net pese 160 Mo et n'a pas toujours a voyager.
"""
import argparse
import os
import shutil
import subprocess
import sys

from . import config

#: jamais synchronise, dans un sens comme dans l'autre
IGNORES = [".DS_Store", "**/.DS_Store", "**/__pycache__/**", "**/*.tmp"]

URI = "hf://buckets/" + config.BUCKET


def _cli():
    """Chemin de la commande hf, ou un message qui dit quoi installer."""
    hf = shutil.which("hf")
    if not hf:
        raise SystemExit(
            "La commande `hf` est introuvable.\n"
            "  uv sync --extra tracking    (puis relancer)\n"
            "  ou : pip install --upgrade huggingface_hub")
    return hf


def sync(sens, dry_run=False, delete=False, poids=True, verbose=False, bucket=None):
    """Lance `hf buckets sync` dans le sens demande.

    sens : "push" (local -> bucket) ou "pull" (bucket -> local).
    """
    hf = _cli()
    uri = "hf://buckets/" + (bucket or config.BUCKET)
    local = config.DATA_ROOT

    if sens == "push" and not os.path.isdir(local):
        raise SystemExit(f"Rien a envoyer : {local} n'existe pas.")
    os.makedirs(local, exist_ok=True)

    source, cible = (local, uri) if sens == "push" else (uri, local)
    cmd = [hf, "buckets", "sync", source, cible, "--ignore-times"]
    for motif in IGNORES:
        cmd += ["--exclude", motif]
    if not poids:
        cmd += ["--exclude", "runs/**/*.pt", "--exclude", "runs/**/*.pth"]
    if delete:
        cmd += ["--delete"]
    if dry_run:
        cmd += ["--dry-run"]
    elif verbose:
        cmd += ["--verbose"]

    print(f"[sync] {source} -> {cible}")
    if delete and not dry_run:
        print("[sync] --delete : les fichiers absents de la source seront "
              "SUPPRIMES de la destination.")
    return subprocess.call(cmd)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sens", choices=["push", "pull"],
                    help="push : local -> bucket. pull : bucket -> local.")
    ap.add_argument("--bucket", default=None,
                    help=f"defaut {config.BUCKET} (ou GPSO_BUCKET)")
    ap.add_argument("--dry-run", action="store_true",
                    help="affiche le plan sans rien transferer")
    ap.add_argument("--delete", action="store_true",
                    help="supprime cote destination ce qui n'existe plus cote "
                         "source ; a n'utiliser qu'apres un --dry-run")
    ap.add_argument("--no-weights", action="store_true",
                    help="n'envoie pas les checkpoints (.pt, .pth) de runs/")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    return sync(args.sens, dry_run=args.dry_run, delete=args.delete,
                poids=not args.no_weights, verbose=args.verbose,
                bucket=args.bucket)


if __name__ == "__main__":
    sys.exit(main())
