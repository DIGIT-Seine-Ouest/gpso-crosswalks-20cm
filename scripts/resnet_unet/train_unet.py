"""
Entrainement d'un U-Net ResNet34 sur le split fige du depot.

POURQUOI CE SCRIPT EXISTE
La baseline du depot vient d'ArcGIS : on a ses poids, mais ni son decoupage
train/val, ni sa boucle d'entrainement. Impossible donc de la rejouer, ni de
garantir qu'elle n'a pas appris sur les tuiles qui servent a la noter.
Ce script comble ce trou : meme architecture que la baseline, memes tuiles,
meme split, meme code de metriques que DINOv3 (models/eval_common.py).

DEUX RECETTES
  --recipe protocole  (defaut)  boucle alignee sur celle de DINOv3 : meme loss,
                                meme augmentation, meme nombre d'epoques, meme
                                selection du checkpoint. C'est la recette a
                                utiliser pour comparer les deux modeles.
  --recipe arcgis               reconstitution de la recette ArcGIS (one-cycle,
                                lr discriminants slice(1.74e-06, 1.74e-05),
                                cross-entropy nue). Sert a verifier qu'on
                                retombe sur la baseline, pas a comparer.

Deux differences avec DINOv3 sont assumees et documentees :
  - la normalisation est ImageNet (celle qu'attend un ResNet34 torchvision),
    pas SAT-493M ; chaque modele recoit le pretraitement pour lequel il a ete
    pre-entraine, ce n'est pas une variable libre ;
  - le lr par defaut est 1e-4 et non 1e-3 : 1e-3 s'applique chez DINOv3 a une
    tete initialisee au hasard, ici il s'appliquerait a un encodeur pre-entraine
    et le detruirait. A confirmer par un balayage sur le volet val.

L'architecture est construite par evaluate_unet.make_unet : exactement celle des
poids ArcGIS, aux conventions fastai v1. Le checkpoint produit ici se relit donc
avec evaluate_unet.py sans conversion.

USAGE
  python models/resnet_unet/train_unet.py --split split_v2.json --epochs 30
  python models/resnet_unet/train_unet.py --recipe arcgis --lr 1.7378e-05
  python models/resnet_unet/train_unet.py --freeze-encoder     # ablation gele/gele
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_common as ec
from evaluate_unet import make_unet

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 512


# ------------------------------------------------------------------ donnees
class TilesTrain(ec.TilesEval):
    """Meme lecture que l'evaluation, plus l'augmentation D4 (flips + rotations
    90 deg) quand augment=True. Strictement la meme augmentation que DINOv3 :
    c'est une des choses qui doivent rester identiques d'un modele a l'autre."""

    def __init__(self, stems, mean, std, augment, root=None):
        super().__init__(stems, mean, std, root)
        self.augment = augment

    def __getitem__(self, i):
        from PIL import Image
        s = self.stems[i]
        img = np.array(Image.open(f"{self.root}/images/{s}.tif").convert("RGB"))
        msk = np.array(Image.open(f"{self.root}/labels/{s}.tif"))
        if self.augment:
            if np.random.rand() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
            if np.random.rand() < 0.5:
                img, msk = img[::-1], msk[::-1]
            k = np.random.randint(4)
            if k:
                img, msk = np.rot90(img, k, (0, 1)), np.rot90(msk, k, (0, 1))
        img = np.ascontiguousarray(img.transpose(2, 0, 1), np.float32) / 255.0
        img = (img - self.mean) / self.std
        msk = np.ascontiguousarray(msk).astype(np.int64)
        return torch.from_numpy(img), torch.from_numpy(msk)


# ------------------------------------------------------------------ loss
def dice_loss(logits, target, eps=1.0):
    """Recopiee a l'identique de train_dinov3.dice_loss (l'importer tirerait
    transformers comme dependance pour rien)."""
    p = logits.softmax(1)[:, 1]
    t = (target == 1).float()
    inter = (p * t).sum()
    return 1.0 - (2 * inter + eps) / (p.sum() + t.sum() + eps)


def build_loss(kind, pos_weight, dev):
    """combo : 0.5 * CE ponderee (classe positive = pos_weight) + 0.5 * Dice
               -> la loss de DINOv3, celle du protocole de comparaison.
       ce    : CrossEntropyLoss nue = CrossEntropyLossFlat, le defaut fastai
               qu'utilise ArcGIS ; rend la valid loss comparable a son log."""
    if kind == "ce":
        ce = nn.CrossEntropyLoss()
        return lambda logits, y: ce(logits, y)
    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], device=dev))
    return lambda logits, y: 0.5 * ce(logits, y) + 0.5 * dice_loss(logits, y)


# ------------------------------------------------------------------ metriques
class SegMetrics:
    """accuracy, Dice et IoU agreges sur TOUT le volet de validation, jamais
    moyennes par batch. Meme convention que train_dino_compare.SegMetrics et que
    la metrique Dice de fastai : les histoires d'entrainement sont superposables."""

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


def hms(sec):
    return f"{int(sec)//3600:02d}:{int(sec)//60%60:02d}:{int(sec)%60:02d}"


# ------------------------------------------------------------------ optim
def param_groups(model):
    """Trois groupes, comme le splitter de fastai pour un unet_learner :
    encodeur bas / encodeur haut / decodeur. Sert aux lr discriminants."""
    body = list(model.layers[0].children()) if hasattr(model, "layers") else []
    cut = len(body) // 2
    enc_lo = [p for m in body[:cut] for p in m.parameters()]
    enc_hi = [p for m in body[cut:] for p in m.parameters()]
    enc_ids = {id(p) for p in enc_lo + enc_hi}
    dec = [p for p in model.parameters() if id(p) not in enc_ids]
    return [g for g in (enc_lo, enc_hi, dec) if g]


def build_optim(model, recipe, lr, epochs, steps_per_epoch):
    """protocole : AdamW + CosineAnnealingLR, un seul lr — le schema DINOv3.
       arcgis    : AdamW + OneCycleLR avec lr discriminants geometriquement
                   espaces entre lr/10 et lr, ce que fait fastai pour un
                   slice(start, stop) reparti sur les groupes de parametres."""
    if recipe == "protocole":
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs), "epoch"
    groups = param_groups(model)
    lrs = np.geomspace(lr / 10, lr, len(groups))
    opt = torch.optim.AdamW([{"params": g, "lr": float(l)} for g, l in zip(groups, lrs)])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[float(l) for l in lrs], epochs=epochs,
        steps_per_epoch=steps_per_epoch, pct_start=0.25)
    return opt, sched, "step"


# ------------------------------------------------------------------ boucle
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=["protocole", "arcgis"], default="protocole")
    ap.add_argument("--split", default=ec.SPLIT)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None,
                    help="defaut : 1e-4 en recette protocole, 1.7378e-05 en recette arcgis")
    ap.add_argument("--loss", choices=["combo", "ce"], default=None,
                    help="defaut : combo en recette protocole, ce en recette arcgis")
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--norm", choices=list(("imagenet", "sat", "none")), default="imagenet")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="ablation : encodeur ResNet34 gele, seul le decodeur apprend "
                         "— le pendant CNN de DINOv3 gele")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--history", default=None)
    args = ap.parse_args()

    if args.lr is None:
        args.lr = 1e-4 if args.recipe == "protocole" else 1.7378e-05
    if args.loss is None:
        args.loss = "combo" if args.recipe == "protocole" else "ce"
    tag = f"_{args.recipe}" if args.recipe != "protocole" else ""
    args.out = args.out or os.path.join(HERE, f"unet_seed{args.seed}{tag}.pt")
    args.history = args.history or os.path.join(HERE, f"history_unet_seed{args.seed}{tag}.json")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = (dev == "cuda") and not args.no_amp

    tr_stems = ec.load_subset(args.split, "train")
    va_stems = ec.load_subset(args.split, "val")
    mean, std = {"imagenet": ec.IMAGENET, "sat": ec.SAT493M,
                 "none": ((0, 0, 0), (1, 1, 1))}[args.norm]

    dl_tr = torch.utils.data.DataLoader(
        TilesTrain(tr_stems, mean, std, True), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, drop_last=True)
    dl_va = torch.utils.data.DataLoader(
        TilesTrain(va_stems, mean, std, False), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers)

    model = make_unet(n_out=2, size=IMG_SIZE, pretrained=True).to(dev)
    if args.freeze_encoder:
        for p in model.layers[0].parameters():
            p.requires_grad_(False)
    n_tot = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)

    loss_fn = build_loss(args.loss, args.pos_weight, dev)
    opt, sched, step_on = build_optim(model, args.recipe, args.lr, args.epochs, len(dl_tr))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    print(f"[data]  {len(tr_stems)} train / {len(va_stems)} val "
          f"({os.path.basename(args.split)}) | norm {args.norm}")
    print(f"[model] U-Net ResNet34 ImageNet | {n_tot/1e6:.1f} M parametres, "
          f"{n_tr/1e6:.1f} M entrainables"
          f"{' (encodeur gele)' if args.freeze_encoder else ''}")
    print(f"[optim] recette {args.recipe} | loss {args.loss} | lr {args.lr:.2e} "
          f"| {args.epochs} epoques | batch {args.batch_size} | seed {args.seed}\n")
    cols = ["epoch", "training loss", "validation loss", "accuracy", "Dice", "IoU", "time"]
    print("".join(c.ljust(21) for c in cols))

    best, hist = -1.0, []
    for ep in range(args.epochs):
        t0 = time.time()
        model.train(); tr_sum = n = 0
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            if step_on == "step":
                sched.step()
            tr_sum += loss.item() * x.size(0); n += x.size(0)
        if step_on == "epoch":
            sched.step()

        model.eval(); m = SegMetrics(); va_sum = vn = 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(dev), y.to(dev)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logits = model(x)
                va_sum += loss_fn(logits.float(), y).item() * x.size(0); vn += x.size(0)
                m.update(logits.argmax(1), y)

        row = [str(ep), str(tr_sum / n), str(va_sum / vn), str(m.accuracy),
               str(m.dice), str(m.iou), hms(time.time() - t0)]
        print("".join(c.ljust(21) for c in row), flush=True)
        hist.append({"epoch": ep, "train_loss": tr_sum / n, "valid_loss": va_sum / vn,
                     "accuracy": m.accuracy, "dice": m.dice, "iou": m.iou})

        # Le checkpoint est choisi sur le volet val, jamais sur test.
        if m.iou > best:
            best = m.iou
            torch.save({"model": model.state_dict(), "epoch": ep, "iou_crosswalk": m.iou,
                        "dice_crosswalk": m.dice, "args": vars(args)}, args.out)

    print(f"\nmeilleur IoU {best:.4f} (epoque "
          f"{max(hist, key=lambda h: h['iou'])['epoch']}) -> {args.out}")
    json.dump({"model": "unet-resnet34-imagenet", "recipe": args.recipe,
               "split": os.path.basename(args.split), "seed": args.seed,
               "freeze_encoder": args.freeze_encoder, "args": vars(args),
               "history": hist, "best_iou": best,
               "best_dice": max(h["dice"] for h in hist)},
              open(args.history, "w"), indent=1)
    print(f"-> {args.history}")


if __name__ == "__main__":
    main()
