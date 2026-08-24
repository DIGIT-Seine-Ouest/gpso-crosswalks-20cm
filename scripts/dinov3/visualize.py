"""
Figures qualitatives et courbes pour le checkpoint DINOv3.
Tourne sur le SET DE VALIDATION uniquement.

Produit dans figures/ :
  qualitative.png   tuiles best / mediane / worst : image, verite terrain, erreurs
  training.png      IoU et Dice par epoque
  errors.png        decomposition des erreurs et balayage de seuil
"""
import argparse, glob, json, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import ndimage
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoImageProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_dinov3 as base
from train_dinov3 import Tiles, Head, extract, _HERE

GREEN, RED, BLUE = (0.13, 0.75, 0.38), (0.90, 0.22, 0.21), (0.20, 0.51, 0.93)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(_HERE, "best_dinov3_head.pt"))
    ap.add_argument("--split", default=base.SPLIT_JSON)
    ap.add_argument("--figdir", default=os.path.join(_HERE, "figures"))
    ap.add_argument("--n-per-group", type=int, default=2)
    ap.add_argument("--min-gt", type=int, default=617,
                    help="px de verite terrain minimum pour qu'une tuile soit illustrable ; "
                         "617 = aire d'un objet median. En dessous ce sont des fragments de "
                         "bord de tuile, dont l'IoU par tuile est degenere.")
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)

    base.SPLIT_JSON = args.split
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(f"{base.ROOT}/images/*.tif"))
    _, va = base.load_split(stems, False)
    print(f"[viz] {len(va)} tuiles de validation")

    proc = AutoImageProcessor.from_pretrained(base.MODEL_ID)
    mean, std = proc.image_mean, proc.image_std
    ds = Tiles(va, mean, std, augment=False)
    backbone = AutoModel.from_pretrained(base.MODEL_ID).to(dev).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    ck = torch.load(args.checkpoint, map_location=dev)
    head = Head(backbone.config.hidden_size).to(dev)
    head.load_state_dict(ck["head"]); head.eval()
    patch = backbone.config.patch_size

    # IoU par tuile pour choisir les exemples
    ious, preds = [], {}
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            xb = x[None].to(dev)
            p = (head(extract(backbone, xb, patch), y.shape[-2:]).softmax(1)[:, 1] > 0.5)[0].cpu().numpy()
            g = (y.numpy() == 1)
            u = (p | g).sum()
            ious.append((p & g).sum() / u if u else np.nan)
            preds[i] = p
    ious = np.array(ious)
    gt_px = np.array([(ds[i][1].numpy() == 1).sum() for i in range(len(ds))])
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
            g = (y.numpy() == 1); p = preds[i]
            tp, fp, fn = p & g, p & ~g, g & ~p

            axes[r, 0].imshow(rgb)
            axes[r, 0].set_ylabel(f"{gname}\n{va[i]}\nIoU {ious[i]:.3f}", fontsize=9)
            if r == 0:
                axes[r, 0].set_title("Orthophoto 20 cm", fontsize=10)

            axes[r, 1].imshow(overlay(rgb, [(contour(g), GREEN), (g, GREEN)], alpha=0.35))
            if r == 0:
                axes[r, 1].set_title("Vérité terrain", fontsize=10)

            axes[r, 2].imshow(overlay(rgb, [(tp, GREEN), (fp, RED), (fn, BLUE)]))
            if r == 0:
                axes[r, 2].set_title("Prédiction", fontsize=10)
            for c in range(3):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            r += 1
    fig.legend(handles=[Patch(color=GREEN, label="vrai positif"),
                        Patch(color=RED, label="faux positif"),
                        Patch(color=BLUE, label="faux négatif")],
               loc="lower center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"DINOv3 SAT-493M gelé — validation, seuil 0,5 (époque {ck['epoch']})\n"
                 f"tuiles contenant au moins un objet complet ; {n_sliver} tuiles à fragment de bord écartées",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.018, 1, 0.985])
    fig.savefig(f"{args.figdir}/qualitative.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] qualitative.png : IoU par tuile med {np.nanmedian(ious):.3f} "
          f"p10 {np.nanpercentile(ious,10):.3f} p90 {np.nanpercentile(ious,90):.3f}")

    # courbe d'entrainement
    h = json.load(open(os.path.join(_HERE, "history_dinov3.json")))["history"]
    ep = [r["epoch"] for r in h]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(ep, [r["iou_crosswalk"] for r in h], color="#2563eb", lw=2, label="IoU passage piéton")
    ax.plot(ep, [r["dice_crosswalk"] for r in h], color="#16a34a", lw=2, ls="--", label="Dice")
    b = max(h, key=lambda r: r["iou_crosswalk"])
    ax.scatter([b["epoch"]], [b["iou_crosswalk"]], color="#dc2626", zorder=5, s=45,
               label=f"meilleure époque {b['epoch']} — IoU {b['iou_crosswalk']:.4f}")
    ax.set_xlabel("époque"); ax.set_ylabel("score (validation)")
    ax.set_title("Entraînement de la tête, backbone gelé", fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{args.figdir}/training.png", dpi=110); plt.close(fig)

    # erreurs + seuil
    m = json.load(open(os.path.join(_HERE, "metrics.json")))
    d = m["decomposition_erreurs"]; sw = m["balayage_seuil"]
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
    a2.set_title("Courbe plate : modèle déjà bien calibré", fontsize=11)
    a2.grid(alpha=0.3); a2.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{args.figdir}/errors.png", dpi=110); plt.close(fig)
    print(f"[viz] figures -> {args.figdir}")


if __name__ == "__main__":
    main()
