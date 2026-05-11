

import argparse
import os
import time

from config import DATASETS, RESULTS_DIR
from dataset import get_dataloaders
from model import get_model
from train import train_model, validate
from evaluate import (
    evaluate_on_test,
    compute_metrics,
    plot_training_curves,
    plot_confusion_matrix,
    plot_misclassified,
)
from utils import set_seed, get_device, save_metrics, make_output_dir, print_summary_table


def run_experiment(dataset_key: str, device):
    """
    Ejecuta el pipeline completo para un dataset:
      1. Carga datos
      2. Crea y entrena el modelo
      3. Evalúa en test
      4. Guarda figuras y métricas
    """
    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_key.upper()}")
    print(f"{'='*60}")

    out_dir = make_output_dir(RESULTS_DIR, dataset_key)

    # ── 1. Datos ──────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_names = get_dataloaders(dataset_key)

    # ── 2. Modelo ─────────────────────────────────────────────
    set_seed()
    model = get_model(device)

    # ── 3. Entrenamiento ──────────────────────────────────────
    t0 = time.time()
    history, _, _ = train_model(model, train_loader, val_loader, device)
    elapsed = time.time() - t0
    print(f"  Entrenamiento completado en {elapsed/60:.1f} min")

    # ── 4. Evaluación en test ─────────────────────────────────
    preds, labels, all_images, all_probs = evaluate_on_test(model, test_loader, device)
    metrics = compute_metrics(preds, labels)
    metrics["training_time_min"] = round(elapsed / 60, 2)

    print(f"  Macro F1 : {metrics['macro_f1']:.4f}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")

    # ── 5. Figuras ────────────────────────────────────────────
    plot_training_curves(
        history,
        save_path=os.path.join(out_dir, "training_curves.png"),
    )
    plot_confusion_matrix(
        preds, labels,
        save_path=os.path.join(out_dir, "confusion_matrix.png"),
    )
    plot_misclassified(
        all_images, all_probs, preds, labels,
        save_path=os.path.join(out_dir, "misclassified.png"),
        n=12,
    )

    # ── 6. Métricas en JSON ───────────────────────────────────
    save_metrics(metrics, os.path.join(out_dir, "metrics.json"))

    # Guardar también el modelo entrenado
    model_path = os.path.join(out_dir, "model.pth")
    import torch
    torch.save(model.state_dict(), model_path)
    print(f"  ✓ Modelo guardado en {model_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Experimento 1 — Baseline CNN")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Dataset a entrenar. 'all' ejecuta los 5 (por defecto).",
    )
    args = parser.parse_args()

    device = get_device()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    keys_to_run = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    all_metrics = {}
    for key in keys_to_run:
        try:
            all_metrics[key] = run_experiment(key, device)
        except FileNotFoundError as e:
            print(f"\n  ✗ Error en dataset '{key}': {e}")
            print("    Revisa BASE_DATA_DIR en config.py\n")

    # ── Tabla resumen final ───────────────────────────────────
    if all_metrics:
        print_summary_table(all_metrics)

        # Guardar tabla en txt
        import json
        summary_path = os.path.join(RESULTS_DIR, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"  Resumen guardado en {summary_path}")


if __name__ == "__main__":
    main()
