"""
DINOv3 SAT-493M gele + tete legere, formate pour comparaison directe
avec le run fastai unet_learner(resnet34).

Reutilise a l'identique le dataset, le backbone et la tete de train_dinov3.py
pour que la seule variable soit l'architecture.

Colonnes loguees, dans l'ordre de fastai :
  epoch | training loss | validation loss | accuracy | Dice | IoU | time
"""
import argparse, glob, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoImageProcessor

import train_dinov3 as base
from train_dinov3 import Tiles, Head, extract, dice_loss, ROOT, MODEL_ID, IMG_SIZE, _HERE


class SegMetrics:
    """accuracy, Dice et IoU agreges sur TOUT le set de validation.

    Dice et IoU portent sur la seule classe passage pieton. L'agregation
    reproduit fastai (Dice accumule inter/union sur le dataset entier,
    ce n'est pas une moyenne des Dice par batch).
    """
    def __init__(self):
        self.inter = self.pred_pos = self.targ_pos = self.correct = self.total = 0

    def update(self, pred, targ):
        p, t = pred == 1, targ == 1
        self.inter    += (p & t).sum().item()
        self.pred_pos += p.sum().item()
        self.targ_pos += t.sum().item()
        self.correct  += (pred == targ).sum().item()
        self.total    += targ.numel()

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


def hms(sec):
    return f"{int(sec)//3600:02d}:{int(sec)//60%60:02d}:{int(sec)%60:02d}"


def build_loss(kind, pos_weight, dev):
    """ce     : CrossEntropyLoss nue = equivalent de CrossEntropyLossFlat (defaut fastai).
                Seul choix qui rend la colonne 'validation loss' comparable au run U-Net.
       combo  : 0.5 * CE ponderee (classe positive = pos_weight) + 0.5 * Dice."""
    if kind == "ce":
        ce = nn.CrossEntropyLoss()
        return lambda logits, y: ce(logits, y)
    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], device=dev))
    return lambda logits, y: 0.5 * ce(logits, y) + 0.5 * dice_loss(logits, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["combo", "ce"], default="combo",
                    help="ce = loss fastai par defaut, rend la validation loss comparable")
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--split", default=base.SPLIT_JSON)
    ap.add_argument("--out", default=os.path.join(_HERE, "best_dino_compare.pt"))
    ap.add_argument("--history", default=os.path.join(_HERE, "history_dino_compare.json"))
    ap.add_argument("--allow-fallback-split", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base.SPLIT_JSON = args.split          # le split doit etre celui du run U-Net

    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(f"{ROOT}/images/*.tif"))
    lab = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{ROOT}/labels/*.tif")}
    stems = [s for s in stems if s in lab]
    if not stems:
        raise SystemExit(f"Aucune paire trouvee sous {ROOT}.")
    tr_stems, va_stems = base.load_split(stems, args.allow_fallback_split)

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    dl_tr = DataLoader(Tiles(tr_stems, proc.image_mean, proc.image_std, True),
                       batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, drop_last=True)
    dl_va = DataLoader(Tiles(va_stems, proc.image_mean, proc.image_std, False),
                       batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    backbone = AutoModel.from_pretrained(MODEL_ID).to(dev).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    patch, dim = backbone.config.patch_size, backbone.config.hidden_size

    head = Head(dim).to(dev)
    loss_fn = build_loss(args.loss, args.pos_weight, dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"backbone gele {sum(p.numel() for p in backbone.parameters())/1e6:.0f} M "
          f"| tete entrainable {sum(p.numel() for p in head.parameters())/1e6:.2f} M "
          f"| loss={args.loss} | {len(tr_stems)} train / {len(va_stems)} val\n")
    cols = ["epoch", "training loss", "validation loss", "accuracy", "Dice", "IoU", "time"]
    print("".join(c.ljust(21) for c in cols))

    best, hist = -1.0, []
    for ep in range(args.epochs):
        t0 = time.time()
        head.train(); tr_sum = n = 0
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            logits = head(extract(backbone, x, patch), y.shape[-2:])
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tr_sum += loss.item() * x.size(0); n += x.size(0)
        sched.step()

        head.eval(); m = SegMetrics(); va_sum = vn = 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(dev), y.to(dev)
                logits = head(extract(backbone, x, patch), y.shape[-2:])
                va_sum += loss_fn(logits, y).item() * x.size(0); vn += x.size(0)
                m.update(logits.argmax(1), y)

        row = [str(ep), str(tr_sum / n), str(va_sum / vn), str(m.accuracy),
               str(m.dice), str(m.iou), hms(time.time() - t0)]
        print("".join(c.ljust(21) for c in row), flush=True)

        hist.append({"epoch": ep, "train_loss": tr_sum / n, "valid_loss": va_sum / vn,
                     "accuracy": m.accuracy, "dice": m.dice, "iou": m.iou})
        if m.iou > best:
            best = m.iou
            torch.save({"head": head.state_dict(), "epoch": ep, "iou_crosswalk": m.iou,
                        "dice_crosswalk": m.dice, "model_id": MODEL_ID, "args": vars(args)},
                       args.out)

    bd = max(h["dice"] for h in hist)
    print(f"\nmeilleur Dice {bd:.4f} | meilleur IoU {best:.4f} -> {args.out}")
    json.dump({"model": "dinov3-vitl16-sat493m-frozen", "loss": args.loss,
               "history": hist, "best_iou": best, "best_dice": bd},
              open(args.history, "w"), indent=1)


if __name__ == "__main__":
    main()
