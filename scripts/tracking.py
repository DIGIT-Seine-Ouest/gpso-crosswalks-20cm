"""Tracage des experiences, optionnel et jamais bloquant.

    python -m scripts.train --model dinov3 --track

Sans `--track`, ou si mlflow n'est pas installe, tout devient muet : le meme
code tourne sur une machine equipee et sur un Colab minimal. C'est la condition
pour que le tracage reste un confort et non une dependance du protocole.

La regle qui gouverne ce module : **une erreur de tracage ne fait jamais echouer
un entrainement**. Un serveur qui ne repond pas a trois heures du matin doit
couter la courbe de la nuit, pas la nuit elle-meme. Toute exception desactive le
tracage, l'annonce une fois, et la boucle continue.

Ou vont les donnees
-------------------
Dans `datasets/mlruns/`, un store de fichiers qu'on consulte avec `mlflow ui`.
Ce chemin n'est pas anodin : `datasets/` est le miroir local du bucket Hugging
Face, si bien que le store voyage avec les donnees. Passer du Mac a Colab tient
alors dans un `python -m scripts.hf_sync pull`.

La variable standard MLFLOW_TRACKING_URI le deplace ailleurs — un serveur
distant, par exemple — sans toucher au code.
"""
import os
import platform
import socket
import subprocess

from . import config

#: nom d'experience par defaut, surchargeable par --experiment
EXPERIENCE = "gpso-crosswalks"

#: store par defaut : un dossier de fichiers sous DATA_ROOT, donc dans le miroir
#: du bucket. MLFLOW_TRACKING_URI le deplace ailleurs sans toucher au code.
STORE = os.environ.get("MLFLOW_TRACKING_URI") or "file://" + config.MLRUNS_DIR


# --------------------------------------------------------------------- contexte
def _commande(args):
    """Sortie d'une commande, ou None si elle echoue. Sert au contexte git."""
    try:
        out = subprocess.run(args, cwd=config.REPO, capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def contexte():
    """Ce qui identifie la machine et la version du code, en tags.

    Sans ca, deux runs aux memes hyperparametres mais aux resultats differents
    restent une enigme : c'est presque toujours le commit ou la machine qui a
    change.
    """
    tags = {
        "machine": socket.gethostname(),
        "plateforme": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
    }
    sha = _commande(["git", "rev-parse", "HEAD"])
    if sha:
        tags["git_commit"] = sha[:12]
        tags["git_propre"] = "non" if _commande(["git", "status", "--porcelain"]) else "oui"
    branche = _commande(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branche:
        tags["git_branche"] = branche
    try:
        import torch
        tags["torch"] = torch.__version__
    except Exception:
        pass
    return tags


def aplatir(obj, prefixe=""):
    """Rend plate une structure imbriquee, en ne gardant que les nombres.

    Le rapport d'evaluation melange des scores, des notes en francais et des
    listes ; seules les grandeurs numeriques sont des metriques.
    """
    plat = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            plat.update(aplatir(v, f"{prefixe}{k}." if prefixe else f"{k}."))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        cle = prefixe.rstrip(".")
        if cle:
            plat[cle] = float(obj)
    return plat


# ----------------------------------------------------------------- modeles
#: definition du modele pyfunc, declaree dans un fichier a part (cf. ce module)
PYFUNC = os.path.join(config.REPO, "scripts", "pyfunc_model.py")


def log_model(tracker, checkpoint, nom_enregistre=None):
    """Attache le checkpoint au run et l'inscrit au registre de modeles.

    Sans cet appel, l'onglet Models de MLflow reste vide : consigner un artefact
    ne suffit pas, le registre demande un modele declare. Comme le reste du
    module, l'echec ne remonte jamais jusqu'a l'appelant.
    """
    if not tracker.actif or not checkpoint or not os.path.exists(checkpoint):
        return None
    try:
        import mlflow

        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=PYFUNC,
            artifacts={"checkpoint": checkpoint},
            code_paths=[os.path.join(config.REPO, "scripts")],
            registered_model_name=nom_enregistre,
            pip_requirements=["torch", "transformers", "numpy", "pillow"],
        )
        print(f"[track] modele enregistre : {nom_enregistre or 'model'} "
              f"({os.path.getsize(checkpoint) / 1e6:.1f} Mo) -> {info.model_uri}",
              flush=True)
        return info.model_uri
    except Exception as e:
        print(f"[track] enregistrement du modele impossible : "
              f"{type(e).__name__} {e}", flush=True)
        return None


# --------------------------------------------------------------------- backends
class Muet:
    """Backend par defaut : accepte tout, n'enregistre rien, ne leve rien."""

    actif = False
    uri = None
    run_id = None

    def log_params(self, params):
        pass

    def log_metrics(self, metrics, step=None):
        pass

    def log_artifact(self, chemin):
        pass

    def set_tags(self, tags):
        pass

    def finish(self, status="FINISHED"):
        pass

    def describe(self):
        return "desactive"


class Mlflow(Muet):
    """Backend reel. Chaque appel est garde : une erreur desactive le tracage."""

    actif = True

    def __init__(self, mlflow, run, uri, experience):
        self._mlflow = mlflow
        self.uri = uri
        self.experience = experience
        self.run_id = run.info.run_id

    def _garde(self, quoi, fn, *a, **kw):
        if not self.actif:
            return
        try:
            fn(*a, **kw)
        except Exception as e:
            self.actif = False
            print(f"[track] {quoi} a echoue, tracage desactive pour la suite "
                  f"du run : {type(e).__name__} {e}", flush=True)

    def log_params(self, params):
        propres = {k: ("" if v is None else v) for k, v in params.items()}
        self._garde("log_params", self._mlflow.log_params, propres)

    def log_metrics(self, metrics, step=None):
        propres = {k: float(v) for k, v in metrics.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if propres:
            self._garde("log_metrics", self._mlflow.log_metrics, propres, step=step)

    def log_artifact(self, chemin):
        if chemin and os.path.exists(chemin):
            self._garde("log_artifact", self._mlflow.log_artifact, chemin)

    def set_tags(self, tags):
        self._garde("set_tags", self._mlflow.set_tags, tags)

    def finish(self, status="FINISHED"):
        if not self.actif:
            return
        try:
            self._mlflow.end_run(status=status)
        except Exception as e:
            print(f"[track] cloture du run impossible : {e}", flush=True)
        self.actif = False

    def describe(self):
        return f"mlflow | experience {self.experience} | run {self.run_id[:12]} | {self.uri}"


# ------------------------------------------------------------------------ start
def start(active, run_name=None, experience=None, tags=None):
    """Ouvre un run, ou renvoie un backend muet.

    Ne leve jamais : si mlflow manque ou si le store est injoignable, on le dit
    et on rend un Muet. L'entrainement demarre dans tous les cas.
    """
    if not active:
        return Muet()
    # MLflow refuse le store fichier depuis la 3.x et invite a passer sur une
    # base. On garde pourtant les fichiers, et deliberement : chaque run y occupe
    # un dossier a identifiant unique, si bien que deux machines qui synchronisent
    # vers le meme bucket fusionnent au lieu de s'ecraser. Une base SQLite est un
    # fichier binaire unique — le Mac et Colab se marcheraient dessus.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import mlflow
    except ImportError:
        print("[track] mlflow n'est pas installe, tracage desactive.\n"
              "        uv sync --extra tracking", flush=True)
        return Muet()
    try:
        uri = STORE
        experience = experience or EXPERIENCE
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experience)
        run = mlflow.start_run(run_name=run_name)
        t = Mlflow(mlflow, run, uri, experience)
        t.set_tags({**contexte(), **(tags or {})})
        return t
    except Exception as e:
        print(f"[track] store injoignable ({type(e).__name__} : {e}), "
              f"tracage desactive.", flush=True)
        return Muet()
