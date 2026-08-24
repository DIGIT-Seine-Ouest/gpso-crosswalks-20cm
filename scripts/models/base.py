"""Interface commune a toutes les architectures.

C'est le seul contrat que connaissent train.py, evaluate.py et visualize.py.
Une architecture qui le respecte est utilisable par tous les outils du depot
sans qu'aucun d'eux ait a la connaitre.

Le contrat tient en quatre points :

  1. forward(x) -> logits de forme (B, 2, H, W), classe 1 = passage pieton ;
  2. NORMALIZATION : la normalisation attendue en entree, propre au modele ;
  3. trainable_parameters() : ce que l'optimiseur doit voir, ce qui exclut un
     eventuel backbone gele ;
  4. trainable_state_dict() / load_trainable_state() : ce qu'on sauvegarde,
     pour ne pas ecrire 300 Mo de poids geles a chaque epoque.
"""
import torch.nn as nn

from .. import config


class SegModel(nn.Module):
    """Base des modeles de segmentation binaire fond / passage pieton."""

    #: identifiant utilise par --model, et enregistre dans les checkpoints
    NAME = "base"
    #: description courte, affichee au lancement et stockee dans les rapports
    DESCRIPTION = ""
    #: normalisation d'entree, celle du pre-entrainement de l'architecture
    NORMALIZATION = config.IMAGENET
    #: taux d'apprentissage par defaut, propre a l'architecture
    DEFAULT_LR = 1e-4

    def forward(self, x):
        """(B, 3, H, W) -> logits (B, 2, H, W)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ optim
    def trainable_parameters(self):
        """Parametres confies a l'optimiseur.

        Par defaut tout ce qui a requires_grad ; un modele a backbone gele n'a
        donc rien a redefinir tant qu'il a bien coupe les gradients.
        """
        return [p for p in self.parameters() if p.requires_grad]

    def param_groups(self):
        """Groupes de parametres pour des taux d'apprentissage discriminants.

        Un seul groupe par defaut. Une architecture encodeur/decodeur peut en
        renvoyer plusieurs, du plus bas niveau au plus haut : la recette
        one-cycle leur attribue alors des lr geometriquement espaces.
        """
        return [self.trainable_parameters()]

    # ------------------------------------------------------------ checkpoints
    def trainable_state_dict(self):
        """Poids a sauvegarder. Par defaut le modele entier."""
        return self.state_dict()

    def load_trainable_state(self, state):
        """Recharge ce qu'a produit trainable_state_dict."""
        self.load_state_dict(state)

    # --------------------------------------------------------------- inference
    def predict_proba(self, x):
        """Probabilite de la classe passage pieton, de forme (B, H, W).

        C'est la forme attendue par metrics.run_eval. Un modele dont la sortie
        n'est pas une paire de logits redefinit cette methode.
        """
        return self(x).softmax(1)[:, 1]

    def describe(self):
        """Ligne d'information affichee au lancement."""
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.trainable_parameters())
        return (f"{self.DESCRIPTION or self.NAME} | {total/1e6:.1f} M parametres, "
                f"{train/1e6:.2f} M entrainables")
