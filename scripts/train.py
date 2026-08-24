"""Entrainement, commun a toutes les architectures.

    python -m scripts.train --model dinov3
    python -m scripts.train --model unet-resnet34
    python -m scripts.train --model unet-resnet50 --epochs 40

Ce script ne connait aucune architecture. Il demande au registre un objet
respectant l'interface SegModel, et se contente de l'entrainer. Tout ce qui doit
rester identique d'un modele a l'autre pour que la comparaison ait un sens vit
ici : les tuiles, l'augmentation, la loss, le nombre d'epoques, la selection du
checkpoint. Tout ce qui est propre a une architecture vit dans son module :
la normalisation d'entree, le taux d'apprentissage par defaut, les groupes de
parametres.

Le checkpoint est choisi sur le volet val, jamais sur test. Le volet test n'est
pas ouvert par ce script.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from . import config, dataset, metrics, models


# ---------------------------------------------------------------------- loss
def dice_loss(logits, target, eps=1.0):
    """Dice sur la seule classe passage pieton.

    Complete la cross-entropy : sur une classe qui occupe 0,67 % des pixels, la
    CE seule converge vers un modele prudent qui sous-segmente.
    """
    p = logits.softmax(1)[:, 1]
    t = (target == 1).float()
    inter = (p * t).sum()
    return 1.0 - (2 * inter + eps) / (p.sum() + t.sum() + eps)


def build_loss(kind, pos_weight, device):
    """combo : 0.5 * CE ponderee + 0.5 * Dice — la loss du protocole de comparaison.
       ce    : CrossEntropyLoss nue, le defaut fastai qu'utilise ArcGIS ; rend la
               validation loss comparable a son journal, mais rien d'autre."""
    if kind == "ce":
        ce = nn.CrossEntropyLoss()
        return lambda logits, y: ce(logits, y)
    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], device=device))
    return lambda logits, y: 0.5 * ce(logits, y) + 0.5 * dice_loss(logits, y)


# --------------------------------------------------------------------- optim
def build_optim(model, recipe, lr, epochs, steps_per_epoch):
    """protocole : AdamW + CosineAnnealingLR, un seul lr. Le schema de reference.
       arcgis    : AdamW + OneCycleLR a lr discriminants geometriquement espaces
                   entre lr/10 et lr, ce que fait fastai pour un slice(a, b)
                   reparti sur les groupes de parametres."""
    if recipe == "protocole":
        opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr)
        return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs), "epoch"
    groups = model.param_groups()
    lrs = np.geomspace(lr / 10, lr, len(groups))
    opt = torch.optim.AdamW([{"params": g, "lr": float(l)} for g, l in zip(groups, lrs)])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[float(l) for l in lrs], epochs=epochs,
        steps_per_epoch=steps_per_epoch, pct_start=0.25)
    return opt, sched, "step"


def hms(sec):
    return f"{int(sec)//3600:02d}:{int(sec)//60%60:02d}:{int(sec)%60:02d}"


# ---------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=models.available(),
                    help="architecture a entrainer")
    ap.add_argument("--recipe", choices=["protocole", "arcgis"], default="protocole",
                    help="protocole : le schema de reference, celui qui rend les "
                         "modeles comparables. arcgis : reconstitution de la recette "
                         "ArcGIS, pour verifier qu'on retombe sur la baseline.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None,
                    help="defaut : le DEFAULT_LR de l'architecture")
    ap.add_argument("--loss", choices=["combo", "ce"], default=None,
                    help="defaut : combo en recette protocole, ce en recette arcgis")
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0,
                    help="graine d'initialisation ; relancer avec 0, 1, 2 donne la "
                         "dispersion entre entrainements identiques")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="ablation : encodeur gele, seul le decodeur apprend")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--data-root", default=None,
                    help=f"racine contenant tiles_train/ val/ test/ (defaut {config.DATA_ROOT})")
    ap.add_argument("--split", default=None,
                    help="HERITAGE : split.json du pilote, decoupage aleatoire qui "
                         "fuit. Ne produit pas un resultat publiable.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--history", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.loss is None:
        args.loss = "combo" if args.recipe == "protocole" else "ce"

    tag = f"_{args.recipe}" if args.recipe != "protocole" else ""
    run_dir = os.path.join(config.RUNS_DIR, args.model)
    os.makedirs(run_dir, exist_ok=True)
    args.out = args.out or os.path.join(run_dir, f"seed{args.seed}{tag}.pt")
    args.history = args.history or os.path.join(run_dir, f"history_seed{args.seed}{tag}.json")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = (device == "cuda") and not args.no_amp

    # Les DataLoader sont construits avant le modele : la normalisation est
    # connue du registre sans avoir a telecharger le moindre poids.
    mean, std = models.normalization(args.model)
    dl_tr = dataset.make_loader("train", mean, std, args.batch_size,
                                workers=args.workers, data_root=args.data_root,
                                split_json=args.split, drop_last=True)
    dl_va = dataset.make_loader("val", mean, std, args.batch_size,
                                workers=args.workers, data_root=args.data_root,
                                split_json=args.split)

    model = models.build(args.model, pretrained=True,
                         freeze_encoder=args.freeze_encoder).to(device)
    if args.lr is None:
        args.lr = model.DEFAULT_LR

    loss_fn = build_loss(args.loss, args.pos_weight, device)
    opt, sched, step_on = build_optim(model, args.recipe, args.lr, args.epochs, len(dl_tr))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    source = os.path.basename(args.split) if args.split else "dossiers tiles_*"
    print(f"[data]  {len(dl_tr.dataset)} train / {len(dl_va.dataset)} val ({source})")
    print(f"[model] {model.describe()}")
    print(f"[optim] recette {args.recipe} | loss {args.loss} | lr {args.lr:.2e} "
          f"| {args.epochs} epoques | batch {args.batch_size} | seed {args.seed}\n")
    cols = ["epoch", "training loss", "validation loss", "accuracy", "Dice", "IoU", "time"]
    print("".join(c.ljust(21) for c in cols))

    best, hist = -1.0, []
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        tr_sum = n = 0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if step_on == "step":
                sched.step()
            tr_sum += loss.item() * x.size(0)
            n += x.size(0)
        if step_on == "epoch":
            sched.step()

        model.eval()
        m = metrics.RunningSeg()
        va_sum = vn = 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logits = model(x)
                va_sum += loss_fn(logits.float(), y).item() * x.size(0)
                vn += x.size(0)
                m.update(logits.argmax(1), y)

        row = [str(ep), str(tr_sum / n), str(va_sum / vn), str(m.accuracy),
               str(m.dice), str(m.iou), hms(time.time() - t0)]
        print("".join(c.ljust(21) for c in row), flush=True)
        hist.append({"epoch": ep, "train_loss": tr_sum / n, "valid_loss": va_sum / vn,
                     "accuracy": m.accuracy, "dice": m.dice, "iou": m.iou})

        # Selection sur le volet val. Le volet test n'est jamais ouvert ici.
        if m.iou > best:
            best = m.iou
            state = model.trainable_state_dict()
            state.update({"model_name": args.model, "epoch": ep,
                          "iou_crosswalk": m.iou, "dice_crosswalk": m.dice,
                          "args": vars(args)})
            torch.save(state, args.out)

    print(f"\nmeilleur IoU {best:.4f} (epoque "
          f"{max(hist, key=lambda h: h['iou'])['epoch']}) -> {args.out}")
    with open(args.history, "w") as f:
        json.dump({"model": args.model, "recipe": args.recipe, "seed": args.seed,
                   "source": source, "freeze_encoder": args.freeze_encoder,
                   "args": vars(args), "history": hist, "best_iou": best,
                   "best_dice": max(h["dice"] for h in hist)}, f, indent=1)
    print(f"-> {args.history}")


if __name__ == "__main__":
    main()
