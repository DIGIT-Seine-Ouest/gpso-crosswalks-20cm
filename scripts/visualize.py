"""Figures qualitatives et cartes d'erreurs, communes a toutes les architectures.

    python -m scripts.visualize --model dinov3 --checkpoint runs/dinov3/seed0.pt

Trois planches :
  qualitative.png  meilleures / medianes / pires tuiles, avec vrais positifs,
                   faux positifs et faux negatifs distingues par couleur ;
  training.png     IoU et Dice par epoque, sur le volet val ;
  errors.png       decomposition des erreurs et balayage du seuil de decision.

Les deux dernieres se construisent a partir des fichiers produits par
scripts.train et scripts.evaluate, elles ne relancent aucun calcul.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from scipy import ndimage

from . import config, dataset, device as devices, models

GREEN, RED, BLUE = (0.13, 0.70, 0.29), (0.86, 0.15, 0.15), (0.15, 0.39, 0.92)


def denorm(x, mean, std):
    a = x.numpy() * np.asarray(std).reshape(3, 1, 1) + np.asarray(mean).reshape(3, 1, 1)
    return np.clip(a.transpose(1, 2, 0), 0, 1)


def overlay(rgb, masks_colors, alpha=0.55):
    out = rgb.copy()
    for m, c in masks_colors:
        out[m] = (1 - alpha) * out[m] + alpha * np.asarray(c)
    return out


def contour(m, w=2):
    return ndimage.binary_dilation(m, np.ones((w * 2 + 1,) * 2, bool)) & ~m


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=models.available())
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--subset", choices=list(config.SUBSETS), default="val")
    ap.add_argument("--figdir", default=None)
    ap.add_argument("--history", default=None, help="defaut : l'historique du run")
    ap.add_argument("--metrics", default=None, help="defaut : le rapport du volet")
    ap.add_argument("--n-per-group", type=int, default=2)
    ap.add_argument("--min-gt", type=int, default=617,
                    help="px de verite terrain minimum pour qu'une tuile soit "
                         "illustrable ; 617 = aire d'un objet median. En dessous ce "
                         "sont des fragments de bord de tuile, dont l'IoU par tuile "
                         "est degenere.")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", choices=list(devices.CHOIX), default=None,
                    help="defaut : le plus rapide present (cuda > mps > cpu)")
    ap.add_argument("--split", default=None, help="HERITAGE : split.json du pilote")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_dir = os.path.join(config.RUNS_DIR, args.model)
    args.figdir = args.figdir or os.path.join(run_dir, "figures")
    args.history = args.history or os.path.join(run_dir, "history_seed0.json")
    args.metrics = args.metrics or os.path.join(run_dir, f"metrics_{args.subset}.json")
    os.makedirs(args.figdir, exist_ok=True)

    device = devices.pick(args.device)
    mean, std = models.normalization(args.model)
    loader = dataset.make_loader(args.subset, mean, std, batch_size=1, augment=False,
                                 shuffle=False, data_root=args.data_root,
                                 split_json=args.split)
    ds, stems = loader.dataset, loader.dataset.stems
    print(f"[viz] {len(ds)} tuiles du volet {args.subset}")

    model = models.build(args.model, pretrained=False)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_trainable_state(state)
    model = model.to(device).eval()
    print(f"[viz] {model.describe()}")

    # IoU par tuile, pour choisir les exemples a illustrer
    ious, preds, gt_px = [], {}, []
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            p = (model.predict_proba(x[None].to(device)) > 0.5)[0].cpu().numpy()
            g = (y.numpy() == 1)
            u = (p | g).sum()
            ious.append((p & g).sum() / u if u else np.nan)
            preds[i] = p
            gt_px.append(int(g.sum()))
    ious, gt_px = np.array(ious), np.array(gt_px)

    n_sliver = int(((gt_px > 0) & (gt_px < args.min_gt)).sum())
    valid = np.where(~np.isnan(ious) & (gt_px >= args.min_gt))[0]
    print(f"[viz] {len(valid)} tuiles illustrables (GT >= {args.min_gt} px) ; "
          f"{n_sliver} ecartees car fragments de bord de tuile")
    order = valid[np.argsort(-ious[valid])]
    k = args.n_per_group
    mid = len(order) // 2
    groups = [("Meilleures", order[:k]),
              ("Médianes", order[mid - k // 2: mid - k // 2 + k]),
              ("Pires", order[-k:])]

    rows = sum(len(idx) for _, idx in groups)
    fig, axes = plt.subplots(rows, 3, figsize=(11.5, 3.9 * rows))
    if rows == 1:
        axes = axes[None]
    r = 0
    for gname, idx in groups:
        for i in idx:
            x, y = ds[i]
            rgb = denorm(x, mean, std)
            g, p = (y.numpy() == 1), preds[i]
            tp, fp, fn = p & g, p & ~g, g & ~p

            axes[r, 0].imshow(rgb)
            axes[r, 0].set_ylabel(f"{gname}\n{stems[i]}\nIoU {ious[i]:.3f}", fontsize=9)
            axes[r, 1].imshow(overlay(rgb, [(contour(g), GREEN), (g, GREEN)], alpha=0.35))
            axes[r, 2].imshow(overlay(rgb, [(tp, GREEN), (fp, RED), (fn, BLUE)]))
            if r == 0:
                for c, t in enumerate(("Orthophoto 20 cm", "Vérité terrain", "Prédiction")):
                    axes[r, c].set_title(t, fontsize=10)
            for c in range(3):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            r += 1
    fig.legend(handles=[Patch(color=GREEN, label="vrai positif"),
                        Patch(color=RED, label="faux positif"),
                        Patch(color=BLUE, label="faux négatif")],
               loc="lower center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"{model.DESCRIPTION} — volet {args.subset}, seuil 0,5 "
                 f"(époque {state.get('epoch')})\n"
                 f"tuiles contenant au moins un objet complet ; "
                 f"{n_sliver} tuiles à fragment de bord écartées", fontsize=12)
    fig.tight_layout(rect=[0, 0.018, 1, 0.985])
    fig.savefig(f"{args.figdir}/qualitative.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] qualitative.png : IoU par tuile med {np.nanmedian(ious):.3f} "
          f"p10 {np.nanpercentile(ious, 10):.3f} p90 {np.nanpercentile(ious, 90):.3f}")

    if os.path.exists(args.history):
        _training_curve(args.history, args.figdir, model.DESCRIPTION)
    else:
        print(f"[viz] historique absent ({args.history}), training.png non produite")
    if os.path.exists(args.metrics):
        _error_figure(args.metrics, args.figdir)
    else:
        print(f"[viz] rapport absent ({args.metrics}), errors.png non produite")
    print(f"[viz] figures -> {args.figdir}")


def _training_curve(path, figdir, title):
    with open(path) as f:
        hist = json.load(f)["history"]
    ep = [r["epoch"] for r in hist]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(ep, [r["iou"] for r in hist], color="#2563eb", lw=2, label="IoU passage piéton")
    ax.plot(ep, [r["dice"] for r in hist], color="#16a34a", lw=2, ls="--", label="Dice")
    b = max(hist, key=lambda r: r["iou"])
    ax.scatter([b["epoch"]], [b["iou"]], color="#dc2626", zorder=5, s=45,
               label=f"meilleure époque {b['epoch']} — IoU {b['iou']:.4f}")
    ax.set_xlabel("époque"); ax.set_ylabel("score (validation)")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{figdir}/training.png", dpi=110); plt.close(fig)


def _error_figure(path, figdir):
    with open(path) as f:
        m = json.load(f)
    d, sw = m["decomposition_erreurs"], m["balayage_seuil"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    lab = ["FP contour", "FN contour", "FP loin", "FN loin"]
    val = [d["fp_contour_pct"], d["fn_contour_pct"], d["fp_loin_pct"], d["fn_loin_pct"]]
    a1.barh(lab[::-1], val[::-1], color=["#94a3b8", "#94a3b8", "#dc2626", "#2563eb"][::-1])
    for i, v in enumerate(val[::-1]):
        a1.text(v + 0.8, i, f"{v:.1f} %", va="center", fontsize=9)
    a1.set_xlim(0, max(val) * 1.25); a1.set_xlabel("part des pixels erronés")
    a1.set_title(f"{d['part_contour_pct']:.0f} % des erreurs sont au contour", fontsize=11)
    a1.grid(axis="x", alpha=0.3)
    ts = sorted(float(t) for t in sw)
    a2.plot(ts, [sw[f"{t}"] for t in ts], color="#2563eb", lw=2)
    bt = m["meilleur_seuil"]
    a2.scatter([bt["seuil"]], [bt["iou"]], color="#dc2626", zorder=5, s=45,
               label=f"optimum {bt['seuil']} — IoU {bt['iou']:.4f}")
    a2.set_xlabel("seuil de décision"); a2.set_ylabel("IoU passage piéton")
    a2.set_title("Sensibilité au seuil de décision", fontsize=11)
    a2.grid(alpha=0.3); a2.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{figdir}/errors.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
