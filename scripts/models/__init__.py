"""Registre des architectures.

L'import est differe : declarer un modele n'oblige pas a installer sa
dependance. On peut evaluer un U-Net sans que transformers soit present, et
entrainer DINOv3 sans que fastai le soit.

Ajouter une architecture demande deux choses, et rien d'autre dans le depot :

  1. un module models/mon_modele.py exposant une constante NORMALIZATION et une
     fonction build(**kwargs) qui renvoie une instance de base.SegModel ;
  2. une ligne dans _REGISTRY ci-dessous.

Un ResNet50 est deja declare : il reutilise models/unet.py avec un autre
encodeur, sans une ligne de code supplementaire. train.py, evaluate.py et
visualize.py n'ont a etre touches dans aucun des deux cas.
"""
import importlib

from .base import SegModel

#: nom passe a --model -> (module qui l'implemente, arguments figes du build)
_REGISTRY = {
    "dinov3": ("scripts.models.dinov3", {}),
    "unet-resnet34": ("scripts.models.unet", {"encoder": "resnet34"}),
    "unet-resnet50": ("scripts.models.unet", {"encoder": "resnet50"}),
}


def available():
    """Noms d'architectures utilisables avec --model."""
    return sorted(_REGISTRY)


def _module(name):
    if name not in _REGISTRY:
        raise SystemExit(f"Modele inconnu : {name!r}\n"
                         f"Disponibles : {', '.join(available())}")
    path, preset = _REGISTRY[name]
    return importlib.import_module(path), preset


def build(name, **kwargs):
    """Instancie une architecture par son nom.

    Les kwargs de l'appelant l'emportent sur les arguments figes du registre ;
    chaque module documente ceux qu'il accepte.
    """
    module, preset = _module(name)
    model = module.build(**{**preset, **kwargs})
    if not isinstance(model, SegModel):
        raise TypeError(f"{_REGISTRY[name][0]}.build() doit renvoyer un SegModel, "
                        f"pas un {type(model).__name__}")
    return model


def normalization(name):
    """Normalisation attendue par une architecture, sans l'instancier.

    Permet de construire les DataLoader avant de charger les poids, qui est
    l'operation lente.
    """
    module, _ = _module(name)
    return module.NORMALIZATION


__all__ = ["SegModel", "available", "build", "normalization"]
