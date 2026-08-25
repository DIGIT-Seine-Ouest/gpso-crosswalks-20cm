"""DINOv3 ViT-L/16 pre-entraine SAT-493M, gele, surmonte d'une tete apprise.

Le backbone n'apprend pas : seule la tete de segmentation est entrainee. C'est
le pari du modele de fondation, ou l'on suppose que des representations apprises
sur 493 millions d'images satellite valent mieux qu'un encodeur reentraine sur
439 passages pietons.

Deux tetes cohabitent, choisies par --model :

  dinov3      tete conv minimale, une seule prise, upsampling bilineaire x16.
              Mesure la qualite des representations du backbone.
  dinov3-dpt  tete DPT : quatre prises a des profondeurs differentes,
              reassemblees a quatre resolutions puis fusionnees en remontant.
              Mesure ce que ces memes representations donnent quand le decodeur
              cesse d'etre le facteur limitant.

Le second existe pour trancher une hypothese ecrite dans le bilan du premier :
sa precision plus faible viendrait de la grille de patchs 16 px, pas de la
semantique du backbone. Meme backbone, meme gel : seul le decodeur change.

Dependance : transformers.
"""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config
from .base import SegModel

MODEL_ID = "facebook/dinov3-vitl16-pretrain-sat493m"

#: normalisation du processor SAT-493M, et non celle d'ImageNet.
#: Recopiee en constante pour pouvoir construire les DataLoader sans telecharger
#: le checkpoint ; identique a AutoImageProcessor.from_pretrained(MODEL_ID).
NORMALIZATION = config.SAT493M

#: profondeurs des quatre prises de la tete DPT, dans hidden_states.
#: L'indice 0 etant les embeddings, ce sont les sorties des blocs 6, 12, 18, 24
#: du ViT-L. Du plus superficiel (texture) au plus profond (semantique).
DPT_LAYERS = (6, 12, 18, 24)


class Head(nn.Module):
    """conv 3x3 -> BN -> ReLU -> conv 1x1, puis reechantillonnage bilineaire.

    Volontairement minimale : plus la tete est legere, plus le score mesure la
    qualite des representations du backbone plutot que la capacite de la tete.
    """

    def __init__(self, in_ch, mid=256, n_cls=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, n_cls, 1),
        )

    def forward(self, feat, out_hw):
        return F.interpolate(self.block(feat), size=out_hw, mode="bilinear",
                             align_corners=False)


# --------------------------------------------------------------------- tete DPT

class ResidualConvUnit(nn.Module):
    """ReLU -> conv3x3 -> BN -> ReLU -> conv3x3 -> BN, plus le skip additif.

    Brique de base du decodeur RefineNet : elle raffine une carte sans changer
    sa resolution ni son nombre de canaux, et la connexion additive laisse le
    gradient traverser la pyramide de fusion sans s'attenuer.
    """

    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        h = self.bn1(self.conv1(self.relu(x)))
        h = self.bn2(self.conv2(self.relu(h)))
        return x + h


class FeatureFusionBlock(nn.Module):
    """Fusionne la carte remontee avec la prise de meme resolution, puis x2.

    C'est l'etage qui distingue la DPT de la tete conv : la resolution est
    regagnee par paliers, chacun reinjectant une prise moins profonde donc plus
    fine. L'upsampling bilineaire final ne fait que x2, jamais x16.
    """

    def __init__(self, ch, with_skip=True):
        super().__init__()
        # L'etage le plus profond n'a rien a fusionner : lui donner un
        # ResidualConvUnit d'entree laisserait 1,18 M de parametres morts dans
        # le checkpoint, et sous le regard de l'optimiseur.
        self.rcu_skip = ResidualConvUnit(ch) if with_skip else None
        self.rcu_out = ResidualConvUnit(ch)

    def forward(self, x, skip=None):
        if skip is not None:
            if self.rcu_skip is None:
                raise RuntimeError("bloc construit sans skip, mais un skip est fourni")
            x = x + self.rcu_skip(skip)
        x = self.rcu_out(x)
        return F.interpolate(x, scale_factor=2.0, mode="bilinear",
                             align_corners=False)


class DPTHead(nn.Module):
    """Reassemblage multi-echelle facon DPT-Large, puis fusion RefineNet.

    Entree : les quatre prises de tokens, chacune (B, g*g, C). Sortie : les
    logits a la resolution de la tuile.

    Les quatre prises sont d'abord ramenees a la meme echelle par une
    LayerNorm chacune. Ce n'est pas cosmetique : sur ce checkpoint, l'ecart-type
    des activations passe de 2,1 au bloc 6 a 40,9 au bloc 24, soit un facteur
    19. Sans cette normalisation la branche profonde ecraserait les trois
    autres, et le reassemblage n'a pas de BatchNorm pour rattraper l'echelle.
    """

    #: canaux intermediaires du reassemblage, une valeur par prise.
    REASSEMBLE_CH = (256, 512, 1024, 1024)

    def __init__(self, in_ch, features=256, n_cls=2):
        super().__init__()
        c1, c2, c3, c4 = self.REASSEMBLE_CH

        # une LayerNorm par prise, appliquee sur les tokens avant la mise en carte
        self.norms = nn.ModuleList([nn.LayerNorm(in_ch) for _ in DPT_LAYERS])

        # projection 1x1 : chaque prise recoit sa propre largeur
        self.proj = nn.ModuleList([
            nn.Conv2d(in_ch, c1, 1),
            nn.Conv2d(in_ch, c2, 1),
            nn.Conv2d(in_ch, c3, 1),
            nn.Conv2d(in_ch, c4, 1),
        ])

        # reechantillonnage vers 1/4, 1/8, 1/16, 1/32 de la tuile.
        # La prise la moins profonde remonte le plus haut : c'est elle qui porte
        # le detail geometrique, et donc le contour du passage pieton.
        self.resize = nn.ModuleList([
            nn.ConvTranspose2d(c1, c1, kernel_size=4, stride=4),
            nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(c4, c4, kernel_size=3, stride=2, padding=1),
        ])

        # mise a la largeur commune du decodeur
        self.layer_rn = nn.ModuleList([
            nn.Conv2d(c, features, 3, padding=1, bias=False)
            for c in (c1, c2, c3, c4)
        ])

        # index 3 = l'etage le plus profond, seul a n'avoir aucun skip
        self.fuse = nn.ModuleList([
            FeatureFusionBlock(features, with_skip=(i != 3)) for i in range(4)
        ])

        half = features // 2
        self.out_conv = nn.Sequential(
            nn.Conv2d(features, half, 3, padding=1, bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
        )
        self.out_final = nn.Sequential(
            nn.Conv2d(half, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_cls, 1),
        )

    def forward(self, tokens, g, out_hw):
        feats = []
        for i, t in enumerate(tokens):
            t = self.norms[i](t)                                  # (B, g*g, C)
            # .contiguous() avant la mise en carte : sans lui, le backward de
            # reshape sur un tenseur transpose echoue sur MPS.
            m = t.transpose(1, 2).contiguous().view(t.shape[0], -1, g, g)
            m = self.resize[i](self.proj[i](m))
            feats.append(self.layer_rn[i](m))
        s1, s2, s3, s4 = feats

        x = self.fuse[3](s4)          # 1/32 -> 1/16
        x = self.fuse[2](x, s3)       # 1/16 -> 1/8
        x = self.fuse[1](x, s2)       # 1/8  -> 1/4
        x = self.fuse[0](x, s1)       # 1/4  -> 1/2

        x = self.out_conv(x)
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return self.out_final(x)


# ------------------------------------------------------------------- segmenteurs

class _DinoV3Base(SegModel):
    """Ce que les deux variantes partagent : le backbone gele et son gel.

    Aucune des deux ne touche aux poids du ViT-L. Tout ce que l'optimiseur voit,
    tout ce qu'un checkpoint contient, c'est la tete.
    """

    NORMALIZATION = NORMALIZATION

    def __init__(self, model_id=MODEL_ID):
        super().__init__()
        from transformers import AutoModel

        self.model_id = model_id
        self.backbone = AutoModel.from_pretrained(model_id)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.patch = self.backbone.config.patch_size

    def train(self, mode=True):
        """Le backbone reste en mode eval meme pendant l'entrainement.

        Il est gele : laisser ses couches de normalisation mettre a jour leurs
        statistiques le ferait deriver silencieusement d'une epoque a l'autre.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def _grid(self, x):
        return x.shape[-1] // self.patch

    def _drop_special(self, t, g):
        """Retire les tokens CLS et registres, deduits de la forme.

        Le nombre de tokens speciaux n'est pas code en dur : ce checkpoint en a
        5 (1 CLS + 4 registres), un autre pourrait en avoir un nombre different.
        """
        n_special = t.shape[1] - g * g
        if n_special < 1:
            raise RuntimeError(f"L={t.shape[1]} < {g*g} tokens de patch")
        return t[:, n_special:, :]

    def trainable_parameters(self):
        return list(self.head.parameters())

    def trainable_state_dict(self):
        return {"head": self.head.state_dict(), "model_id": self.model_id}

    def load_trainable_state(self, state):
        # Accepte aussi bien le format produit ici que le "head" nu des
        # checkpoints de la Phase 1.
        self.head.load_state_dict(state["head"] if "head" in state else state)

    def describe(self):
        frozen = sum(p.numel() for p in self.backbone.parameters())
        train = sum(p.numel() for p in self.trainable_parameters())
        return (f"{self.DESCRIPTION} | tete {train/1e6:.2f} M entrainables, "
                f"backbone gele {frozen/1e6:.0f} M")


class DinoV3Segmenter(_DinoV3Base):
    """Une seule prise, la derniere, et une tete de deux convolutions."""

    NAME = "dinov3"
    DESCRIPTION = "DINOv3 ViT-L/16 SAT-493M gele + tete conv"
    DEFAULT_LR = 1e-3          # s'applique a une tete initialisee au hasard

    def __init__(self, model_id=MODEL_ID, mid=256, n_cls=2):
        super().__init__(model_id=model_id)
        self.head = Head(self.backbone.config.hidden_size, mid, n_cls)

    @torch.no_grad()
    def _extract(self, x):
        """(B,3,H,W) -> (B,C,H/patch,W/patch)."""
        g = self._grid(x)
        # Le backbone tourne en fp16 sur CUDA, ou la precision mixte est acquise.
        # Ailleurs — MPS, CPU — on laisse la precision du tenseur d'entree : c'est
        # la boucle d'entrainement qui decide, via scripts/device.py.
        with (torch.autocast("cuda", dtype=torch.float16) if x.is_cuda
              else contextlib.nullcontext()):
            lhs = self.backbone(pixel_values=x).last_hidden_state       # (B, L, C)
        t = self._drop_special(lhs, g).float()                          # (B, g*g, C)
        return t.transpose(1, 2).reshape(x.shape[0], -1, g, g).contiguous()

    def forward(self, x):
        return self.head(self._extract(x), x.shape[-2:])


class DinoV3DPTSegmenter(_DinoV3Base):
    """Quatre prises, reassemblees et fusionnees en remontant la resolution."""

    NAME = "dinov3-dpt"
    DESCRIPTION = "DINOv3 ViT-L/16 SAT-493M gele + tete DPT"
    DEFAULT_LR = 3e-4          # decodeur profond : le 1e-3 de la tete conv diverge

    def __init__(self, model_id=MODEL_ID, features=256, n_cls=2):
        super().__init__(model_id=model_id)
        self.features = features
        self.head = DPTHead(self.backbone.config.hidden_size, features, n_cls)

    @torch.no_grad()
    def _extract(self, x):
        """(B,3,H,W) -> quatre tenseurs de tokens (B, g*g, C).

        Les prises sont les sorties brutes des blocs, avant la layernorm finale
        du backbone : les quatre sont donc homogenes entre elles, ce qui est ce
        que le reassemblage attend. Leur mise a l'echelle est faite par la tete.
        """
        g = self._grid(x)
        with (torch.autocast("cuda", dtype=torch.float16) if x.is_cuda
              else contextlib.nullcontext()):
            hs = self.backbone(pixel_values=x,
                               output_hidden_states=True).hidden_states
        return [self._drop_special(hs[i], g).float() for i in DPT_LAYERS]

    def forward(self, x):
        return self.head(self._extract(x), self._grid(x), x.shape[-2:])

    def describe(self):
        return f"{super().describe()} | features {self.features}"


def build(model_id=MODEL_ID, head="conv", mid=256, features=256, n_cls=2, **_):
    if head == "conv":
        return DinoV3Segmenter(model_id=model_id, mid=mid, n_cls=n_cls)
    if head == "dpt":
        return DinoV3DPTSegmenter(model_id=model_id, features=features,
                                  n_cls=n_cls)
    raise SystemExit(f"Tete inconnue : {head!r} (attendu 'conv' ou 'dpt')")
