"""
Socle d'evaluation partage par tous les modeles du depot.

Le but est qu'un seul code calcule les metriques, quel que soit le modele : un ecart
entre deux lignes du tableau comparatif vient alors des modeles, jamais de la mesure.

Chaque modele fournit seulement une fonction predict(x) -> probabilite de la classe
positive, de forme (B, H, W), et la normalisation qu'il attend en entree.
"""
import json, os
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TILES = os.environ.get("GPSO_TILES", os.path.join(REPO, "tiles"))
SPLIT = os.environ.get("GPSO_SPLIT", os.path.join(REPO, "split.json"))

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
SAT493M  = ((0.430, 0.411, 0.296), (0.213, 0.156, 0.143))


def load_split(path=None):
    path = path or SPLIT
    with open(path) as f:
        s = json.load(f)
    return s["train"], s["val"]


def load_subset(path=None, name="val"):
    """Retourne un sous-ensemble nomme du split : train, val, ou test s'il existe.

    Permet d'appliquer exactement les memes metriques aux trois volets, ce qui est
    la condition pour comparer honnetement deux entrainements."""
    path = path or SPLIT
    with open(path) as f:
        s = json.load(f)
    if name not in s or not isinstance(s[name], list):
        dispo = [k for k, v in s.items() if isinstance(v, list)]
        raise SystemExit(f"'{name}' absent de {os.path.basename(path)} ; disponibles : {dispo}")
    return s[name]


class TilesEval(Dataset):
    """Lecture deterministe, sans augmentation. La normalisation est propre au modele."""

    def __init__(self, stems, mean, std, root=None):
        self.stems, self.root = stems, root or TILES
        self.mean = np.asarray(mean, np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        s = self.stems[i]
        img = np.array(Image.open(f"{self.root}/images/{s}.tif").convert("RGB"))
        msk = np.array(Image.open(f"{self.root}/labels/{s}.tif"))
        img = np.ascontiguousarray(img.transpose(2, 0, 1), np.float32) / 255.0
        img = (img - self.mean) / self.std
        return torch.from_numpy(img), torch.from_numpy(np.ascontiguousarray(msk).astype(np.int64))


def d4_tta(predict, x):
    """Moyenne des probabilites sur les 8 symetries du carre."""
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
    """Accumule les compteurs sur tout le set. Rien n'est moyenne par batch :
    intersection et union sont sommees globalement, comme la metrique Dice de fastai."""

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


def run_eval(predict, stems, mean, std, batch_size=2, tta=True, root=None, device="cuda"):
    """Evalue un modele sur `stems`. predict(x) -> proba classe positive (B, H, W)."""
    dl = DataLoader(TilesEval(stems, mean, std, root), batch_size=batch_size,
                    shuffle=False, num_workers=0)
    ev = SegEvaluator()
    ev_tta = SegEvaluator() if tta else None
    with torch.no_grad():
        for x, y in dl:
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
            "note": "moyenne des probabilites sur les 8 symetries du carre",
        }
    return rep


def pretty(rep, titre):
    p, o, d = rep["pixel"], rep["objet"], rep["decomposition_erreurs"]
    lo, hi = p["iou_crosswalk_ci95"]
    print(f"\n--- {titre} ---")
    print(f"tuiles {rep['n_tiles']} | objets {rep['n_objects']} | seuil {rep['threshold']}")
    print(f"IoU passage pieton   {p['iou_crosswalk']:.4f}   IC95 [{lo:.4f}, {hi:.4f}]")
    print(f"Dice / F1            {p['dice_f1_crosswalk']:.4f}")
    print(f"Precision            {p['precision_crosswalk']:.4f}")
    print(f"Rappel               {p['recall_crosswalk']:.4f}")
    print(f"Accuracy globale     {p['accuracy_global']:.4f}  (tout-fond = {p['accuracy_baseline_tout_fond']:.4f})")
    print(f"Detection objet      {o['detectes_couverture_50pct']}/{o['total']} = {o['taux_detection']*100:.1f}%")
    print(f"Erreurs au contour   {d['part_contour_pct']:.1f}% des {d['total_px']} px errones")
    print(f"Meilleur seuil       {rep['meilleur_seuil']['seuil']} -> IoU {rep['meilleur_seuil']['iou']:.4f}")
    if "tta_d4" in rep:
        t = rep["tta_d4"]
        print(f"TTA D4               IoU {t['iou_seuil_0.5']:.4f} a 0.5 | "
              f"{t['iou_meilleur_seuil']:.4f} au seuil {t['meilleur_seuil']}")
