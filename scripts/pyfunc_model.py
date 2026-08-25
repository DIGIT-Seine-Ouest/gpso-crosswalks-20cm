"""Definition du modele pyfunc charge par MLflow.

Fichier separe et volontairement : MLflow recommande de declarer un modele
depuis un script plutot que de serialiser un objet Python avec cloudpickle, qui
execute du code arbitraire au chargement. Ce module n'est jamais importe par le
reste du paquet — seul MLflow le lit, au moment d'enregistrer puis de recharger.

Ne transporte que le checkpoint, c'est-a-dire la tete pour un modele a backbone
gele : neuf megaoctets pour dinov3 au lieu de mille deux cents. Le backbone est
reconstruit depuis Hugging Face au chargement, exactement comme a
l'entrainement. Le registre stocke ce qui a ete appris, pas ce qui a ete
telecharge — mais il faut donc un acces reseau la premiere fois.
"""
import mlflow
import numpy as np
import torch

try:                                  # depuis le depot
    from scripts import models
except ImportError:                   # depuis les code_paths du modele MLflow
    import models


class Segmenteur(mlflow.pyfunc.PythonModel):
    """Entree  : (B, 3, H, W) float32 deja normalise pour le modele.
       Sortie  : (B, H, W) float32, probabilite de la classe passage pieton."""

    def load_context(self, context):
        etat = torch.load(context.artifacts["checkpoint"], map_location="cpu",
                          weights_only=False)
        self.nom = etat.get("model_name", "dinov3")
        self.model = models.build(self.nom, pretrained=True)
        self.model.load_trainable_state(etat)
        self.model.eval()

    def predict(self, context, model_input, params=None):
        x = torch.from_numpy(np.asarray(model_input, dtype="float32"))
        if x.ndim == 3:
            x = x[None]
        with torch.no_grad():
            return self.model.predict_proba(x).numpy()


mlflow.models.set_model(Segmenteur())
