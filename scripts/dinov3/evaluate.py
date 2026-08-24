"""
Evaluation du checkpoint DINOv3 sur le SET DE VALIDATION uniquement.

Les metriques viennent de models/eval_common.py, partage avec les autres modeles du
depot : un ecart dans le tableau comparatif vient des modeles, jamais de la mesure.
"""
import argparse, json, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import eval_common as ec
from train_dinov3 import Head, extract, MODEL_ID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(HERE, "best_dinov3_head.pt"))
    ap.add_argument("--split", default=ec.SPLIT)
    ap.add_argument("--subset", choices=["train", "val", "test"], default="val",
                    help="volet du split a evaluer ; memes metriques pour les trois")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out = args.out or os.path.join(
        HERE, "metrics.json" if args.subset == "val" else f"metrics_{args.subset}.json")
    va = ec.load_subset(args.split, args.subset)
    print(f"[eval] {len(va)} tuiles du volet {args.subset.upper()} de {os.path.basename(args.split)}")

    backbone = AutoModel.from_pretrained(MODEL_ID).to(dev).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    patch = backbone.config.patch_size
    ck = torch.load(args.checkpoint, map_location=dev)
    head = Head(backbone.config.hidden_size).to(dev)
    head.load_state_dict(ck["head"]); head.eval()
    print(f"[eval] checkpoint epoque {ck['epoch']}, backbone gele")

    def predict(x):
        return head(extract(backbone, x, patch), x.shape[-2:]).softmax(1)[:, 1]

    mean, std = ec.SAT493M                      # stats du processor SAT-493M, pas ImageNet
    rep = ec.run_eval(predict, va, mean, std, batch_size=args.batch_size,
                      tta=not args.no_tta, device=dev)
    rep.update({
        "model": "DINOv3 ViT-L/16 SAT-493M gele + tete conv",
        "checkpoint": os.path.basename(args.checkpoint),
        "epoch": ck["epoch"],
        "normalisation": "sat493m",
        "evaluated_on": f"volet {args.subset} de {os.path.basename(args.split)}",
        "subset": args.subset,
        "reserve": ("split.json a servi a selectionner le checkpoint (meilleure epoque). "
                    "Ces scores sont donc legerement optimistes : il n'y a pas de set de "
                    "test tenu a l'ecart."),
    })
    json.dump(rep, open(args.out, "w"), indent=1, ensure_ascii=False)
    ec.pretty(rep, f"DINOv3 SAT-493M gele — {args.subset} de {os.path.basename(args.split)}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
