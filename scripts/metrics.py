"""Metriques de segmentation, communes a tous les modeles.

Un seul code calcule les scores, quelle que soit l'architecture : un ecart entre
deux lignes du tableau comparatif vient alors des modeles, jamais de la mesure.

Deux niveaux, correspondant au cadrage du cahier de recherche (chapitre 01) :

  - pixel  : IoU, Dice, precision, rappel. Metriques d'optimisation du reseau,
             utilisees pour choisir l'epoque du checkpoint.
  - objet  : taux de detection a 50 % de couverture. Metrique metier, celle qui
             dit combien de passages pietons du terrain ont ete retrouves.

L'accuracy globale n'est calculee que pour etre mise en regard de la baseline
tout-fond : sur une classe representant 0,67 % des pixels, elle ne discrimine
rien et n'est publiee qu'a ce titre.

Toutes les grandeurs sont accumulees sur l'ensemble du volet, jamais moyennees
par batch : intersection et union sont sommees globalement.
"""
import numpy as np
import torch
from scipy import ndimage


class RunningSeg:
    """Accuracy, Dice et IoU agreges, pour le suivi epoque par epoque.

    Volontairement leger : appele a chaque epoque d'entrainement, il ne fait
    aucune analyse morphologique. Le rapport complet est produit par
    SegEvaluator, une seule fois, en fin de course.
    """

    def __init__(self):
        self.inter = self.pred_pos = self.targ_pos = self.correct = self.total = 0

    def update(self, pred, targ):
        p, t = pred == 1, targ == 1
        self.inter += (p & t).sum().item()
        self.pred_pos += p.sum().item()
        self.targ_pos += t.sum().item()
        self.correct += (pred == targ).sum().item()
        self.total += targ.numel()

    @property
    def accuracy(self):
        return self.correct / self.total if self.total else float("nan")

    @property
    def dice(self):
        d = self.pred_pos + self.targ_pos
        return 2 * self.inter / d if d else float("nan")

    @property
    def iou(self):
        u = self.pred_pos + self.targ_pos - self.inter
        return self.inter / u if u else float("nan")


def d4_tta(predict, x):
    """Moyenne des probabilites sur les 8 symetries du carre.

    Une orthophoto n'ayant pas d'orientation privilegiee, moyenner les huit
    predictions stabilise le resultat sans rien changer au modele.
    """
    acc = 0
    for k in range(4):
        xr = torch.rot90(x, k, (2, 3))
        for flip in (False, True):
            xi = torch.flip(xr, (3,)) if flip else xr
            p = predict(xi)
            if flip:
                p = torch.flip(p, (2,))
            acc = acc + torch.rot90(p, -k, (1, 2))
    return acc / 8


class SegEvaluator:
    """Rapport complet sur un volet : pixel, objet, decomposition des erreurs."""

    def __init__(self, thresholds=None, main=0.5):
        self.ths = [round(float(t), 2) for t in (thresholds if thresholds is not None
                                                 else np.arange(0.05, 0.96, 0.05))]
        self.main = main
        self.sweep = {t: {"i": 0, "u": 0, "tp": 0, "fp": 0, "fn": 0} for t in self.ths}
        self.per_tile = []
        self.fp_b = self.fp_f = self.fn_b = self.fn_f = 0
        self.comp_tot = self.comp_hit = 0
        self.correct = self.total = self.tn = 0

    def update(self, prob, gt):
        for t in self.ths:
            pd = prob > t
            s = self.sweep[t]
            tp = (pd & gt).sum().item()
            s["i"] += tp
            s["u"] += (pd | gt).sum().item()
            s["tp"] += tp
            s["fp"] += (pd & ~gt).sum().item()
            s["fn"] += (~pd & gt).sum().item()

        pd = prob > self.main
        self.correct += (pd == gt).sum().item()
        self.total += gt.numel()
        self.tn += (~pd & ~gt).sum().item()
        for i in range(gt.shape[0]):
            g = gt[i].cpu().numpy()
            p = pd[i].cpu().numpy()
            self.per_tile.append((int((p & g).sum()), int((p | g).sum()), int(g.sum())))
            band = (ndimage.binary_dilation(g, np.ones((5, 5), bool)) &
                    ~ndimage.binary_erosion(g, np.ones((5, 5), bool)))
            fp, fn = p & ~g, g & ~p
            self.fp_b += int((fp & band).sum()); self.fp_f += int((fp & ~band).sum())
            self.fn_b += int((fn & band).sum()); self.fn_f += int((fn & ~band).sum())
            lab, n = ndimage.label(g, np.ones((3, 3), int))
            self.comp_tot += n
            for k in range(1, n + 1):
                m = lab == k
                if (p & m).sum() > 0.5 * m.sum():
                    self.comp_hit += 1

    def report(self):
        s = self.sweep[self.main]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        arr = np.array(self.per_tile, dtype=np.int64)
        rng = np.random.RandomState(0)
        boot = []
        for _ in range(2000):
            idx = rng.randint(0, len(arr), len(arr))
            u = arr[idx, 1].sum()
            boot.append(arr[idx, 0].sum() / u if u else np.nan)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        err = self.fp_b + self.fp_f + self.fn_b + self.fn_f
        best_t = max(self.sweep, key=lambda t: self.sweep[t]["i"] / max(self.sweep[t]["u"], 1))
        pt_iou = arr[:, 0] / np.maximum(arr[:, 1], 1)
        return {
            "n_tiles": len(arr),
            "n_objects": self.comp_tot,
            "threshold": self.main,
            "pixel": {
                "iou_crosswalk": s["i"] / s["u"],
                "iou_crosswalk_ci95": [float(lo), float(hi)],
                "dice_f1_crosswalk": 2 * tp / (2 * tp + fp + fn),
                "precision_crosswalk": tp / (tp + fp) if tp + fp else 0.0,
                "recall_crosswalk": tp / (tp + fn) if tp + fn else 0.0,
                "iou_background": self.tn / (self.total - tp),
                "accuracy_global": self.correct / self.total,
                "accuracy_baseline_tout_fond": 1 - (tp + fn) / self.total,
            },
            "objet": {
                "detectes_couverture_50pct": self.comp_hit,
                "total": self.comp_tot,
                "taux_detection": self.comp_hit / self.comp_tot if self.comp_tot else float("nan"),
                "manques": self.comp_tot - self.comp_hit,
                "note": ("comptage par tuile : un passage pieton visible sur plusieurs "
                         "tuiles chevauchantes est compte plusieurs fois. Le comptage "
                         "par objet du terrain demande le recollage en Lambert-93."),
            },
            "iou_par_tuile": {
                "mediane": float(np.median(pt_iou)),
                "p10": float(np.percentile(pt_iou, 10)),
                "p90": float(np.percentile(pt_iou, 90)),
            },
            "decomposition_erreurs": {
                "total_px": err,
                "fp_contour_pct": 100 * self.fp_b / err, "fp_loin_pct": 100 * self.fp_f / err,
                "fn_contour_pct": 100 * self.fn_b / err, "fn_loin_pct": 100 * self.fn_f / err,
                "part_contour_pct": 100 * (self.fp_b + self.fn_b) / err,
                "note": "contour = bande de +/-2 px autour du perimetre de la verite terrain",
            },
            "balayage_seuil": {str(t): self.sweep[t]["i"] / self.sweep[t]["u"] for t in self.ths},
            "meilleur_seuil": {"seuil": best_t,
                               "iou": self.sweep[best_t]["i"] / self.sweep[best_t]["u"]},
        }


def run_eval(predict, loader, tta=True, device="cuda"):
    """Evalue un modele sur un DataLoader.

    predict(x) -> probabilite de la classe passage pieton, de forme (B, H, W).
    C'est la seule chose que run_eval sait d'un modele.
    """
    ev = SegEvaluator()
    ev_tta = SegEvaluator() if tta else None
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            gt = (y == 1)
            ev.update(predict(x), gt)
            if ev_tta is not None:
                ev_tta.update(d4_tta(predict, x), gt)
    rep = ev.report()
    if ev_tta is not None:
        t = ev_tta.report()
        rep["tta_d4"] = {
            "iou_seuil_0.5": t["pixel"]["iou_crosswalk"],
            "meilleur_seuil": t["meilleur_seuil"]["seuil"],
            "iou_meilleur_seuil": t["meilleur_seuil"]["iou"],
            "dice_f1": t["pixel"]["dice_f1_crosswalk"],
            "taux_detection": t["objet"]["taux_detection"],
            "note": "moyenne des probabilites sur les 8 symetries du carre",
        }
    return rep


def pretty(rep, titre):
    """Rendu console d'un rapport."""
    p, o, d = rep["pixel"], rep["objet"], rep["decomposition_erreurs"]
    lo, hi = p["iou_crosswalk_ci95"]
    print(f"\n--- {titre} ---")
    print(f"tuiles {rep['n_tiles']} | objets {rep['n_objects']} | seuil {rep['threshold']}")
    print(f"IoU passage pieton   {p['iou_crosswalk']:.4f}   IC95 [{lo:.4f}, {hi:.4f}]")
    print(f"Dice / F1            {p['dice_f1_crosswalk']:.4f}")
    print(f"Precision            {p['precision_crosswalk']:.4f}")
    print(f"Rappel               {p['recall_crosswalk']:.4f}")
    print(f"Accuracy globale     {p['accuracy_global']:.4f}  "
          f"(tout-fond = {p['accuracy_baseline_tout_fond']:.4f})")
    print(f"Detection objet      {o['detectes_couverture_50pct']}/{o['total']} = "
          f"{o['taux_detection']*100:.1f}%")
    print(f"Erreurs au contour   {d['part_contour_pct']:.1f}% des {d['total_px']} px errones")
    print(f"Meilleur seuil       {rep['meilleur_seuil']['seuil']} -> "
          f"IoU {rep['meilleur_seuil']['iou']:.4f}")
    if "tta_d4" in rep:
        t = rep["tta_d4"]
        print(f"TTA D4               IoU {t['iou_seuil_0.5']:.4f} a 0.5 | "
              f"{t['iou_meilleur_seuil']:.4f} au seuil {t['meilleur_seuil']}")
