"""Evaluation, commune a toutes les architectures.

    python -m scripts.evaluate --model dinov3 --checkpoint runs/dinov3/seed0.pt
    python -m scripts.evaluate --model unet-resnet34 --weights resnet_unet.pth
    python -m scripts.evaluate --model dinov3 --checkpoint ... --subset test

Les metriques viennent de scripts/metrics.py, partage par tous les modeles : un
ecart dans le tableau comparatif vient des modeles, jamais de la mesure.

Regle du protocole : le volet test ne s'evalue qu'une fois, a la fin, apres que
la meilleure epoque a ete choisie sur val. Passer --subset test avant d'avoir
fige le checkpoint revient a s'entrainer sur le test.
"""
import argparse
import json
import os

import torch

from . import config, dataset, device as devices, metrics, models, tracking


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=models.available())
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint produit par scripts.train")
    ap.add_argument("--weights", default=None,
                    help="poids externes a charger a la construction "
                         "(le .pth ArcGIS, par exemple)")
    ap.add_argument("--subset", choices=list(config.SUBSETS), default="val",
                    help="volet a evaluer ; memes metriques pour les trois")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--device", choices=list(devices.CHOIX), default=None,
                    help="defaut : le plus rapide present (cuda > mps > cpu)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default=None, help="HERITAGE : split.json du pilote")
    ap.add_argument("--out", default=None)
    ap.add_argument("--track", action="store_true",
                    help="trace l'evaluation dans MLflow, en run distinct de "
                         "l'entrainement et tague par volet")
    ap.add_argument("--experiment", default=tracking.EXPERIENCE)
    ap.add_argument("--run-name", default=None,
                    help="defaut : eval_{modele}_{volet}")
    args = ap.parse_args(argv)
    if not args.checkpoint and not args.weights:
        ap.error("il faut --checkpoint (entraine ici) ou --weights (poids externes)")
    return args


def main(argv=None):
    args = parse_args(argv)
    device = devices.pick(args.device)

    run_dir = os.path.join(config.RUNS_DIR, args.model)
    os.makedirs(run_dir, exist_ok=True)
    args.out = args.out or os.path.join(run_dir, f"metrics_{args.subset}.json")

    mean, std = models.normalization(args.model)
    loader = dataset.make_loader(args.subset, mean, std, args.batch_size,
                                 augment=False, shuffle=False,
                                 data_root=args.data_root, split_json=args.split)
    source = os.path.basename(args.split) if args.split else config.SUBSET_DIRS[args.subset]
    print(f"[eval] {len(loader.dataset)} tuiles du volet {args.subset.upper()} ({source})")

    model = models.build(args.model, pretrained=not args.weights, weights=args.weights)
    epoch = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        declared = state.get("model_name")
        if declared and declared != args.model:
            raise SystemExit(f"Le checkpoint declare le modele {declared!r}, "
                             f"mais --model vaut {args.model!r}.")
        model.load_trainable_state(state)
        epoch = state.get("epoch")
    model = model.to(device).eval()
    print(f"[eval] {model.describe()}"
          + (f" | checkpoint epoque {epoch}" if epoch is not None else ""))

    run_name = args.run_name or f"eval_{args.model}_{args.subset}"
    track = tracking.start(args.track, run_name=run_name, experience=args.experiment,
                           tags={"modele": args.model, "volet": args.subset,
                                 "type": "evaluation", "phase": "2"})
    empreinte = dataset.fingerprint(args.subset, args.data_root, args.split)
    track.log_params({
        "model": args.model, "subset": args.subset, "device": device,
        "tta": not args.no_tta, "batch_size": args.batch_size,
        "checkpoint": os.path.basename(args.checkpoint) if args.checkpoint else None,
        "weights": os.path.basename(args.weights) if args.weights else None,
        "epoch": epoch, "n_tiles": len(loader.dataset),
        f"empreinte_{args.subset}": empreinte[:16],
    })
    print(f"[track] {track.describe()}")
    print(f"[eval]  empreinte {args.subset} {empreinte[:12]}")

    rep = metrics.run_eval(model.predict_proba, loader,
                           tta=not args.no_tta, device=device)
    rep.update({
        "model": args.model,
        "description": model.DESCRIPTION,
        "checkpoint": os.path.basename(args.checkpoint) if args.checkpoint else None,
        "weights": os.path.basename(args.weights) if args.weights else None,
        "epoch": epoch,
        "subset": args.subset,
        "evaluated_on": source,
    })
    if args.weights and os.path.basename(args.weights) == "resnet_unet.pth":
        rep["reserve_fuite"] = (
            "ATTENTION : ces poids viennent d'ArcGIS, entraine sur son propre "
            "decoupage aleatoire des 582 tuiles du pilote, que le paquet .dlpk "
            "n'enregistre pas. Une partie inconnue des tuiles d'evaluation a donc "
            "probablement servi a son entrainement. Ce score est optimiste et n'est "
            "PAS comparable a celui d'un modele entraine par scripts.train sur le "
            "decoupage spatial.")

    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)

    # aplatir : le rapport melange scores, notes en francais et listes ; seules
    # les grandeurs numeriques sont des metriques.
    track.log_metrics(tracking.aplatir(rep))
    track.log_artifact(args.out)
    track.finish()
    metrics.pretty(rep, f"{model.DESCRIPTION} — volet {args.subset} ({source})")
    if "reserve_fuite" in rep:
        print("\n!! " + rep["reserve_fuite"])
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
