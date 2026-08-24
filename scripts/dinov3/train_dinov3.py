"""
Segmentation de passages pietons - DINOv3 SAT-493M gele + tete legere.
Dataset : /content/gpso-crosswalks-20cm/tiles  (582 tuiles 512x512, ~0.67% pos)
"""
import argparse, glob, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor

_HERE     = os.path.dirname(os.path.abspath(__file__))
REPO      = os.path.abspath(os.path.join(_HERE, "..", ".."))   # racine du depot
ROOT      = os.environ.get("GPSO_TILES", os.path.join(REPO, "tiles"))
SPLIT_JSON = os.environ.get("GPSO_SPLIT", os.path.join(REPO, "split.json"))
MODEL_ID  = "facebook/dinov3-vitl16-pretrain-sat493m"
IMG_SIZE  = 512

# ---------------------------------------------------------------- split
def load_split(stems, allow_fallback):
    """Retourne (train_stems, val_stems).

    >>> SEULE FONCTION A MODIFIER quand le split de reference est fourni. <<<
    Format attendu de split.json : {"train": ["000000000000", ...], "val": [...]}
    """
    if os.path.exists(SPLIT_JSON):
        with open(SPLIT_JSON) as f:
            s = json.load(f)
        tr = [x for x in s["train"]]
        va = [x for x in s["val"]]
        known = set(stems)
        miss = [x for x in tr + va if x not in known]
        if miss:
            raise ValueError(f"{len(miss)} stems du split absents du dataset, ex: {miss[:3]}")
        print(f"[split] charge depuis {SPLIT_JSON} : {len(tr)} train / {len(va)} val")
        return tr, va
    if not allow_fallback:
        raise SystemExit(
            f"Aucun split trouve a {SPLIT_JSON}.\n"
            "Fournis-le au format {\"train\": [stems], \"val\": [stems]},\n"
            "ou relance avec --allow-fallback-split pour un 80/20 seed 42 (ecrit dans le JSON)."
        )
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(stems))
    cut = int(0.8 * len(stems))
    tr = sorted(stems[i] for i in idx[:cut])
    va = sorted(stems[i] for i in idx[cut:])
    with open(SPLIT_JSON, "w") as f:
        json.dump({"train": tr, "val": va, "note": "fallback 80/20 seed 42"}, f, indent=1)
    print(f"[split] FALLBACK 80/20 seed 42 ecrit dans {SPLIT_JSON} : {len(tr)} train / {len(va)} val")
    return tr, va

# ---------------------------------------------------------------- data
class Tiles(Dataset):
    def __init__(self, stems, mean, std, augment):
        self.stems, self.augment = stems, augment
        self.mean = np.asarray(mean, np.float32).reshape(3, 1, 1)
        self.std  = np.asarray(std,  np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        s = self.stems[i]
        img = np.array(Image.open(f"{ROOT}/images/{s}.tif").convert("RGB"))      # HWC uint8
        msk = np.array(Image.open(f"{ROOT}/labels/{s}.tif"))                     # HW uint16
        if self.augment:
            if np.random.rand() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]        # flip horizontal
            if np.random.rand() < 0.5:
                img, msk = img[::-1], msk[::-1]              # flip vertical
            k = np.random.randint(4)                          # rotation 90 * k
            if k:
                img, msk = np.rot90(img, k, (0, 1)), np.rot90(msk, k, (0, 1))
        img = np.ascontiguousarray(img.transpose(2, 0, 1), np.float32) / 255.0
        img = (img - self.mean) / self.std
        msk = np.ascontiguousarray(msk).astype(np.int64)      # uint16 -> int64 pour la CE
        return torch.from_numpy(img), torch.from_numpy(msk)

# ---------------------------------------------------------------- modele
class Head(nn.Module):
    """conv 3x3 -> BN -> ReLU -> conv 1x1, puis upsampling bilineaire vers 512."""
    def __init__(self, in_ch, mid=256, n_cls=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, n_cls, 1),
        )

    def forward(self, feat, out_hw):
        return F.interpolate(self.block(feat), size=out_hw, mode="bilinear", align_corners=False)

@torch.no_grad()
def extract(backbone, x, patch):
    """(B,3,H,W) -> (B,C,H/patch,W/patch). Le nombre de tokens speciaux est deduit,
    pas code en dur : 1 CLS + 4 registers pour ce checkpoint (1029 - 32*32 = 5)."""
    g = x.shape[-1] // patch
    with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
        lhs = backbone(pixel_values=x).last_hidden_state          # (B, L, C)
    n_special = lhs.shape[1] - g * g
    assert n_special >= 1, f"L={lhs.shape[1]} < {g*g} patch tokens"
    t = lhs[:, n_special:, :].float()                              # (B, g*g, C)
    return t.transpose(1, 2).reshape(x.shape[0], -1, g, g).contiguous()

# ---------------------------------------------------------------- loss & metrique
def dice_loss(logits, target, eps=1.0):
    p = logits.softmax(1)[:, 1]                    # proba classe passage pieton
    t = (target == 1).float()
    inter = (p * t).sum()
    return 1.0 - (2 * inter + eps) / (p.sum() + t.sum() + eps)

class IoUPositive:
    """IoU de la SEULE classe 1, agregee sur tout le set (pas une moyenne de batches)."""
    def __init__(self):
        self.inter = self.union = 0
    def update(self, pred, target):
        p, t = pred == 1, target == 1
        self.inter += (p & t).sum().item()
        self.union += (p | t).sum().item()
    def value(self):
        return self.inter / self.union if self.union else float("nan")

# ---------------------------------------------------------------- boucle
def main():
    global SPLIT_JSON
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--split", default=SPLIT_JSON,
                    help="split fige a utiliser ; le meme fichier pour tous les modeles")
    ap.add_argument("--seed", type=int, default=0,
                    help="graine d'initialisation ; relancer avec 0, 1, 2 donne la "
                         "dispersion entre entrainements identiques")
    ap.add_argument("--out", default=None)
    ap.add_argument("--history", default=None)
    ap.add_argument("--allow-fallback-split", action="store_true")
    args = ap.parse_args()
    suffix = "" if args.seed == 0 else f"_seed{args.seed}"
    args.out = args.out or os.path.join(_HERE, f"best_dinov3_head{suffix}.pt")
    args.history = args.history or os.path.join(_HERE, f"history_dinov3{suffix}.json")

    SPLIT_JSON = args.split
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(f"{ROOT}/images/*.tif"))
    lab = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{ROOT}/labels/*.tif")}
    stems = [s for s in stems if s in lab]        # appariement strict par nom
    print(f"[data] {len(stems)} paires image/label appariees")
    if not stems:
        raise SystemExit(f"Aucune paire trouvee sous {ROOT} - verifie le chemin.")

    tr_stems, va_stems = load_split(stems, args.allow_fallback_split)

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    mean, std = proc.image_mean, proc.image_std   # stats SAT-493M, pas ImageNet
    print(f"[norm] mean={tuple(round(m,3) for m in mean)} std={tuple(round(s,3) for s in std)}")

    dl_tr = DataLoader(Tiles(tr_stems, mean, std, True), batch_size=args.batch_size,
                       shuffle=True, num_workers=args.workers, drop_last=True)
    dl_va = DataLoader(Tiles(va_stems, mean, std, False), batch_size=args.batch_size,
                       shuffle=False, num_workers=args.workers)

    backbone = AutoModel.from_pretrained(MODEL_ID).to(dev).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    patch, dim = backbone.config.patch_size, backbone.config.hidden_size

    with torch.no_grad():                          # controle des shapes, une fois
        probe = backbone(pixel_values=torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=dev)).last_hidden_state
    g = IMG_SIZE // patch
    print(f"[backbone] last_hidden_state {tuple(probe.shape)} | grille {g}x{g}={g*g} "
          f"| tokens speciaux {probe.shape[1]-g*g} (1 CLS + {backbone.config.num_register_tokens} registers) "
          f"| slice [:, {probe.shape[1]-g*g}:, :]")
    del probe

    head = Head(dim).to(dev)
    n_par = sum(p.numel() for p in head.parameters())
    print(f"[head] {n_par/1e6:.2f} M parametres entrainables "
          f"(backbone gele : {sum(p.numel() for p in backbone.parameters())/1e6:.0f} M)")

    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, args.pos_weight], device=dev))
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, hist = -1.0, []
    for ep in range(1, args.epochs + 1):
        head.train(); tot = 0.0; t0 = time.time()
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            feat = extract(backbone, x, patch)
            logits = head(feat, y.shape[-2:])
            loss = 0.5 * ce(logits, y) + 0.5 * dice_loss(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
        sched.step()

        head.eval(); m = IoUPositive()
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(dev), y.to(dev)
                pred = head(extract(backbone, x, patch), y.shape[-2:]).argmax(1)
                m.update(pred, y)
        iou = m.value()
        hist.append(iou)
        flag = ""
        if iou > best:
            best = iou
            torch.save({"head": head.state_dict(), "epoch": ep, "iou_crosswalk": iou,
                        "model_id": MODEL_ID, "args": vars(args)}, args.out)
            flag = "  <- best, sauvegarde"
        print(f"ep {ep:2d}/{args.epochs} | loss {tot/max(len(dl_tr.dataset),1):.4f} "
              f"| IoU passage pieton {iou:.4f} | lr {sched.get_last_lr()[0]:.2e} "
              f"| {time.time()-t0:.0f}s{flag}", flush=True)

    print(f"\nMeilleur IoU classe passage pieton : {best:.4f}  ->  {args.out}")
    json.dump({"iou_crosswalk_per_epoch": hist, "best": best,
               "split": os.path.basename(args.split), "seed": args.seed,
               "args": vars(args)},
              open(args.history, "w"), indent=1)

if __name__ == "__main__":
    main()
