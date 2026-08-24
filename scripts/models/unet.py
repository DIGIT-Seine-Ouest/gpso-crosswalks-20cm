"""U-Net a encodeur ResNet, construit avec fastai DynamicUnet.

Deux usages, qui partagent la meme architecture :

  - `unet-resnet34` avec `arcgis_tail=True` reproduit exactement le reseau des
    poids exportes par ArcGIS (UnetClassifier 2.4). C'est la baseline du depot :
    on a ses poids mais ni son decoupage train/val ni sa boucle, on ne peut donc
    ni la rejouer ni garantir qu'elle n'a pas appris sur ses propres tuiles de
    notation. Pouvoir reconstruire son architecture au bit pres permet au moins
    de la reentrainer proprement sur le decoupage spatial du depot.

  - tout autre encodeur (`resnet50`...) utilise la queue standard de fastai v2 :
    rien a reproduire, le reseau est entraine de zero sur nos tuiles.

Dependance : fastai, torchvision.
"""
import torch
import torch.nn as nn

from .. import config
from .base import SegModel

NORMALIZATION = config.IMAGENET

#: encodeurs disponibles -> (constructeur torchvision, enum de poids)
ENCODERS = {
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "resnet34": ("resnet34", "ResNet34_Weights"),
    "resnet50": ("resnet50", "ResNet50_Weights"),
    "resnet101": ("resnet101", "ResNet101_Weights"),
}


def make_unet(encoder="resnet34", n_out=2, size=config.IMG_SIZE, pretrained=False,
              arcgis_tail=False):
    """Construit le reseau, sans charger de poids de segmentation.

    arcgis_tail=True remplace la queue de DynamicUnet par celle des conventions
    fastai v1, seule facon de charger le state_dict d'ArcGIS en mode strict :
      - PixelShuffle_ICNR : `shuf.conv.0` en v1, `shuf.0.0` en v2 ;
      - la queue v1 enchaine MergeLayer -> res_block en goulot (99->49->99) ->
        conv 1x1, sans le ResizeToOrig ni le ToTensorBase ajoutes en v2.
    Ce chemin n'est valide que pour un encodeur ResNet34 : le goulot a 99 canaux
    depend des largeurs de cet encodeur precis.
    """
    from fastai.vision.all import (ConvLayer, DynamicUnet, MergeLayer, SequentialEx,
                                   create_body)
    import torchvision.models as tvm

    if encoder not in ENCODERS:
        raise SystemExit(f"Encodeur inconnu : {encoder!r} "
                         f"(disponibles : {', '.join(ENCODERS)})")
    fn_name, weights_name = ENCODERS[encoder]
    weights = getattr(tvm, weights_name).IMAGENET1K_V1 if pretrained else None
    body = create_body(getattr(tvm, fn_name)(weights=weights), n_in=3,
                       pretrained=pretrained, cut=-2)
    net = DynamicUnet(body, n_out, (size, size))

    if arcgis_tail:
        if encoder != "resnet34":
            raise SystemExit("arcgis_tail n'est valide que pour resnet34 : le goulot "
                             "a 99 canaux depend des largeurs de cet encodeur.")
        ni = 99                                # 96 (PixelShuffle) + 3 (image d'origine)
        net.layers = nn.ModuleList(list(net.layers)[:9] + [
            MergeLayer(dense=True),
            SequentialEx(ConvLayer(ni, ni // 2, norm_type=None),
                         ConvLayer(ni // 2, ni, norm_type=None),
                         MergeLayer()),
            ConvLayer(ni, n_out, ks=1, act_cls=None, norm_type=None),
        ])
    return net


class UNetResNet(SegModel):
    NAME = "unet-resnet34"
    DESCRIPTION = "U-Net ResNet34 ImageNet"
    NORMALIZATION = NORMALIZATION
    DEFAULT_LR = 1e-4          # s'applique a un encodeur pre-entraine, pas a une
                               # tete neuve : 1e-3 le detruirait

    def __init__(self, encoder="resnet34", n_cls=2, size=config.IMG_SIZE,
                 pretrained=True, arcgis_tail=False, freeze_encoder=False):
        super().__init__()
        self.encoder_name = encoder
        self.arcgis_tail = arcgis_tail
        self.NAME = f"unet-{encoder}"
        self.DESCRIPTION = f"U-Net {encoder} ImageNet"
        self.net = make_unet(encoder, n_cls, size, pretrained, arcgis_tail)
        if freeze_encoder:
            for p in self.net.layers[0].parameters():
                p.requires_grad_(False)
            self.DESCRIPTION += " (encodeur gele)"

    def forward(self, x):
        return self.net(x)

    def param_groups(self):
        """Encodeur bas / encodeur haut / decodeur, comme le splitter de fastai.

        Sert aux taux d'apprentissage discriminants de la recette arcgis. Les
        groupes vides sont ecartes, ce qui gere le cas de l'encodeur gele.
        """
        body = list(self.net.layers[0].children()) if hasattr(self.net, "layers") else []
        cut = len(body) // 2
        enc_lo = [p for m in body[:cut] for p in m.parameters() if p.requires_grad]
        enc_hi = [p for m in body[cut:] for p in m.parameters() if p.requires_grad]
        seen = {id(p) for p in enc_lo + enc_hi}
        dec = [p for p in self.parameters() if p.requires_grad and id(p) not in seen]
        return [g for g in (enc_lo, enc_hi, dec) if g]

    def load_pretrained_weights(self, path):
        """Charge des poids de segmentation existants dans cette architecture.

        Accepte indifferemment un state_dict nu aux conventions fastai v1 (le
        .pth d'ArcGIS) et un checkpoint produit par train.py. Le chargement est
        strict : aucune clef manquante ni surnumeraire, donc l'architecture est
        bien celle des poids.
        """
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and isinstance(sd.get("model"), dict):
            print(f"[model] checkpoint train.py : epoque {sd.get('epoch')}, "
                  f"IoU val {sd.get('iou_crosswalk')}")
            sd = sd["model"]
        renamed = {k.replace("shuf.conv.", "shuf.0.")
                    .replace("layers.8.conv.", "layers.8.0."): v
                   for k, v in sd.items()}
        missing, unexpected = self.net.load_state_dict(renamed, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"chargement non strict : {len(missing)} manquants, "
                f"{len(unexpected)} surnumeraires -> {list(missing)[:3]} "
                f"{list(unexpected)[:3]}")
        return self

    def trainable_state_dict(self):
        return {"model": self.net.state_dict(), "encoder": self.encoder_name}

    def load_trainable_state(self, state):
        self.net.load_state_dict(state["model"] if "model" in state else state)


def build(encoder="resnet34", n_cls=2, size=config.IMG_SIZE, pretrained=True,
          arcgis_tail=False, freeze_encoder=False, weights=None, **_):
    """weights : chemin d'un .pth a charger apres construction (poids ArcGIS ou
    checkpoint du depot). Force arcgis_tail pour un ResNet34 charge depuis un
    fichier, sans quoi le state_dict ne peut pas s'appliquer."""
    if weights and encoder == "resnet34":
        arcgis_tail = True
    model = UNetResNet(encoder=encoder, n_cls=n_cls, size=size,
                       pretrained=pretrained and not weights,
                       arcgis_tail=arcgis_tail, freeze_encoder=freeze_encoder)
    if weights:
        model.load_pretrained_weights(weights)
    return model
