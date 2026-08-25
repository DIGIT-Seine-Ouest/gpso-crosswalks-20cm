"""Inscrit au registre MLflow un checkpoint deja produit.

    python -m scripts.register_model --model dinov3

Sert au rattrapage : un run termine avant que l'enregistrement existe, ou un
checkpoint recu d'une autre machine. Les entrainements lances depuis
`scripts.train` le font desormais tout seuls, en fin de course.

Le run d'origine est rouvert et complete, plutot que doublonne : les metriques
et le modele restent dans la meme fiche.
"""
import argparse
import os
import sys

from . import config, models, tracking


def dernier_run(experience, run_name):
    """Identifiant du run le plus recent portant ce nom, ou None."""
    import mlflow

    mlflow.set_tracking_uri(tracking.STORE)
    exp = mlflow.get_experiment_by_name(experience)
    if exp is None:
        return None
    df = mlflow.search_runs([exp.experiment_id],
                            filter_string=f"tags.mlflow.runName = '{run_name}'",
                            order_by=["start_time DESC"], max_results=1)
    return None if df.empty else df.iloc[0]["run_id"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=models.available())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recipe", default="protocole")
    ap.add_argument("--checkpoint", default=None,
                    help="defaut : runs/{modele}/seed{graine}.pt")
    ap.add_argument("--experiment", default=tracking.EXPERIENCE)
    ap.add_argument("--run-id", default=None,
                    help="defaut : le dernier run portant le nom attendu")
    ap.add_argument("--name", default=None,
                    help="nom dans le registre ; defaut gpso-{modele}")
    args = ap.parse_args(argv)

    tag = f"_{args.recipe}" if args.recipe != "protocole" else ""
    ckpt = args.checkpoint or os.path.join(config.RUNS_DIR, args.model,
                                           f"seed{args.seed}{tag}.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"Checkpoint introuvable : {ckpt}")

    try:
        import mlflow
    except ImportError:
        raise SystemExit("mlflow n'est pas installe.\n  uv sync --extra tracking")

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    run_name = f"{args.model}_{args.recipe}_seed{args.seed}"
    run_id = args.run_id or dernier_run(args.experiment, run_name)
    if not run_id:
        raise SystemExit(f"Aucun run nomme {run_name!r} dans l'experience "
                         f"{args.experiment!r}. Preciser --run-id.")

    mlflow.set_tracking_uri(tracking.STORE)
    print(f"[reg] run {run_id[:12]} ({run_name})")
    print(f"[reg] checkpoint {ckpt} ({os.path.getsize(ckpt)/1e6:.1f} Mo)")

    with mlflow.start_run(run_id=run_id):
        t = tracking.Mlflow(mlflow, mlflow.active_run(), tracking.STORE,
                            args.experiment)
        t.log_artifact(ckpt)
        uri = tracking.log_model(t, ckpt, args.name or f"gpso-{args.model}")
    return 0 if uri else 1


if __name__ == "__main__":
    sys.exit(main())
