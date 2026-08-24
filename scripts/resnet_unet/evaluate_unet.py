"""
Evaluation du U-Net ResNet34 (paquet ArcGIS) sur le SET DE VALIDATION de split.json.

Utilise models/eval_common.py : exactement le meme code de metriques que DINOv3,
pour que l'ecart entre les deux lignes du comparatif vienne des modeles seuls.

Le .pth exporte par ArcGIS 2.4 est un state_dict de DynamicUnet suivant les conventions
fastai v1. Avec fastai 2.x deux choses different et sont corrigees ici :
  - PixelShuffle_ICNR : `shuf.conv.0` en v1, `shuf.0.0` en v2 (simple renommage) ;
  - la queue du reseau : v1 enchaine MergeLayer -> res_block en goulot (99->49->99)
    -> conv 1x1, sans le ResizeToOrig ni le ToTensorBase ajoutes en v2.
La reconstruction ci-dessous charge le state_dict en mode strict, sans clef manquante
ni surnumeraire : l'architecture est donc bien celle des poids.
"""
import argparse, json, os, sys, warnings
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_common as ec

HERE = os.path.dirname(os.path.abspath(__file__))
NORMS = {"imagenet": ec.IMAGENET, "sat": ec.SAT493M, "none": ((0, 0, 0), (1, 1, 1))}


def make_unet(n_out=2, size=512, pretrained=False):
    """Construit l'architecture exacte des poids ArcGIS, sans les charger.

    Partagee avec train_unet.py : le modele entraine ici et le modele evalue la
    sont le meme objet, aucun risque d'ecart d'architecture entre les deux.
    pretrained=True part de l'encodeur ResNet34 ImageNet (entrainement) ;
    False laisse les poids au hasard (ils vont etre ecrases par le state_dict)."""
    from fastai.vision.all import create_body, DynamicUnet, ConvLayer, MergeLayer, SequentialEx
    from torchvision.models import resnet34, ResNet34_Weights

    w = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    body = create_body(resnet34(weights=w), n_in=3, pretrained=pretrained, cut=-2)
    m = DynamicUnet(body, n_out, (size, size))
    ni = 99                                   # 96 (PixelShuffle) + 3 (image d'origine)
    m.layers = nn.ModuleList(list(m.layers)[:9] + [
        MergeLayer(dense=True),
        SequentialEx(ConvLayer(ni, ni // 2, norm_type=None),
                     ConvLayer(ni // 2, ni, norm_type=None),
                     MergeLayer()),
        ConvLayer(ni, n_out, ks=1, act_cls=None, norm_type=None),
    ])
    return m


def build_unet(pth, n_out=2, size=512):
    """Charge un state_dict dans cette architecture. Accepte indifferemment les
    poids ArcGIS (state_dict nu, conventions fastai v1) et un checkpoint produit
    par train_unet.py (dict avec une clef "model")."""
    sd = torch.load(pth, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and isinstance(sd.get("model"), dict):
        print(f"[eval] checkpoint train_unet.py : epoque {sd.get('epoch')}, "
              f"IoU val {sd.get('iou_crosswalk')}")
        sd = sd["model"]
    m = make_unet(n_out, size, pretrained=False)
    ren = {k.replace("shuf.conv.", "shuf.0.").replace("layers.8.conv.", "layers.8.0."): v
           for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(ren, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"chargement non strict : {len(missing)} manquants, "
                           f"{len(unexpected)} surnumeraires -> {missing[:3]} {unexpected[:3]}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(HERE, "resnet_unet.pth"))
    ap.add_argument("--split", default=ec.SPLIT)
    ap.add_argument("--norm", choices=list(NORMS), default="imagenet",
                    help="normalisation attendue par le modele ; ArcGIS utilise "
                         "imagenet pour de l'imagerie 3 bandes")
    ap.add_argument("--subset", choices=["train", "val", "test"], default="val",
                    help="volet du split a evaluer ; memes metriques pour les trois")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out = args.out or os.path.join(
        HERE, "metrics.json" if args.subset == "val" else f"metrics_{args.subset}.json")
    va = ec.load_subset(args.split, args.subset)
    print(f"[eval] {len(va)} tuiles du volet {args.subset.upper()} de {os.path.basename(args.split)}")

    model = build_unet(args.weights).to(dev).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[eval] U-Net ResNet34 charge en strict, {n/1e6:.1f} M parametres, "
          f"normalisation {args.norm}")

    def predict(x):
        return model(x).softmax(1)[:, 1]

    mean, std = NORMS[args.norm]
    rep = ec.run_eval(predict, va, mean, std, batch_size=args.batch_size,
                      tta=not args.no_tta, device=dev)
    arcgis = os.path.basename(args.weights) == "resnet_unet.pth"
    rep.update({
        "model": ("U-Net ResNet34 (ArcGIS UnetClassifier 2.4)" if arcgis
                  else "U-Net ResNet34 (train_unet.py)"),
        "weights": os.path.basename(args.weights),
        "normalisation": args.norm,
        "evaluated_on": f"volet {args.subset} de {os.path.basename(args.split)}",
        "subset": args.subset,
    })
    if arcgis:
        # La reserve ne vaut que pour les poids ArcGIS : un modele entraine par
        # train_unet.py connait son split, il n'y a rien a signaler.
        rep["reserve_fuite"] = (
            "ATTENTION : ce modele a ete entraine par ArcGIS sur son propre decoupage "
            "aleatoire des 582 tuiles, que le paquet .dlpk n'enregistre pas. Une partie "
            "inconnue des tuiles de validation de split.json a donc probablement servi a "
            "son entrainement. Ce score est optimiste et n'est PAS directement comparable "
            "a celui de DINOv3, qui n'a jamais vu ces tuiles. Pour un comparatif honnete, "
            "reentrainer un U-Net avec train_unet.py sur le meme split.")
    json.dump(rep, open(args.out, "w"), indent=1, ensure_ascii=False)
    ec.pretty(rep, f"U-Net ResNet34 — {args.subset} de {os.path.basename(args.split)}")
    if arcgis:
        print("\n!! " + rep["reserve_fuite"])
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
