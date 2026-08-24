"""DINOv3 ViT-L/16 pre-entraine SAT-493M, gele, surmonte d'une tete legere.

Le backbone n'apprend pas : seule la tete de segmentation est entrainee. C'est
le pari du modele de fondation, ou l'on suppose que des representations apprises
sur 493 millions d'images satellite valent mieux qu'un encodeur reentraine sur
439 passages pietons.

Dependance : transformers.
"""
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


class DinoV3Segmenter(SegModel):
    NAME = "dinov3"
    DESCRIPTION = "DINOv3 ViT-L/16 SAT-493M gele + tete conv"
    NORMALIZATION = NORMALIZATION
    DEFAULT_LR = 1e-3          # s'applique a une tete initialisee au hasard

    def __init__(self, model_id=MODEL_ID, mid=256, n_cls=2):
        super().__init__()
        from transformers import AutoModel

        self.model_id = model_id
        self.backbone = AutoModel.from_pretrained(model_id)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.patch = self.backbone.config.patch_size
        self.head = Head(self.backbone.config.hidden_size, mid, n_cls)

    def train(self, mode=True):
        """Le backbone reste en mode eval meme pendant l'entrainement.

        Il est gele : laisser ses couches de normalisation mettre a jour leurs
        statistiques le ferait deriver silencieusement d'une epoque a l'autre.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def _extract(self, x):
        """(B,3,H,W) -> (B,C,H/patch,W/patch).

        Le nombre de tokens speciaux est deduit de la forme de la sortie, pas
        code en dur : ce checkpoint en a 5 (1 CLS + 4 registers), un autre
        pourrait en avoir un nombre different.
        """
        g = x.shape[-1] // self.patch
        with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
            lhs = self.backbone(pixel_values=x).last_hidden_state       # (B, L, C)
        n_special = lhs.shape[1] - g * g
        if n_special < 1:
            raise RuntimeError(f"L={lhs.shape[1]} < {g*g} tokens de patch")
        t = lhs[:, n_special:, :].float()                               # (B, g*g, C)
        return t.transpose(1, 2).reshape(x.shape[0], -1, g, g).contiguous()

    def forward(self, x):
        return self.head(self._extract(x), x.shape[-2:])

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


def build(model_id=MODEL_ID, mid=256, n_cls=2, **_):
    return DinoV3Segmenter(model_id=model_id, mid=mid, n_cls=n_cls)
