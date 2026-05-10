import argparse
import os
import time
import json
import torch
import config

from config import DATASETS, RESULTS_DIR
from dataset import get_dataloaders
from model import get_model
from train import train_model
from evaluate import (
    evaluate_on_test,
    compute_metrics,
    plot_training_curves,
    plot_confusion_matrix,
    plot_misclassified,
)
from utils import set_seed, get_device, save_metrics, make_output_dir, print_summary_table


def save_used_config(out_dir: str):
    used_config = {
        "EXPERIMENT_NAME": config.EXPERIMENT_NAME,
        "OPTIMIZER_NAME": config.OPTIMIZER_NAME,
        "LOSS_NAME": config.LOSS_NAME,
        "USE_AUGMENTATION": config.USE_AUGMENTATION,
        "USE_CONV_DROPOUT": config.USE_CONV_DROPOUT,
        "DROPOUT_CONV_P": config.DROPOUT_CONV_P,
        "DROPOUT_FC_P": config.DROPOUT_FC_P,
        "WEIGHT_DECAY": config.WEIGHT_DECAY,
        "LEARNING_RATE": config.LEARNING_RATE,
        "MOMENTUM": config.MOMENTUM,
        "LABEL_SMOOTHING": config.LABEL_SMOOTHING,
        "FOCAL_GAMMA": config.FOCAL_GAMMA,
        "BATCH_SIZE": config.BATCH_SIZE,
        "EPOCHS": config.EPOCHS,
    }

    with open(os.path.join(out_dir, "config_used.json"), "w") as f:
        json.dump(used_config, f, indent=2)


def run_experiment(dataset_key: str, device):
    print(f"\n{'=' * 60}")
    print(f"  DATASET: {dataset_key.upper()}")
    print(f"{'=' * 60}")

    out_dir = make_output_dir(RESULTS_DIR, dataset_key)
    save_used_config(out_dir)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(dataset_key)

    set_seed()
    model = get_model(device)

    t0 = time.time()
    history, _, _ = train_model(model, train_loader, val_loader, device)
    elapsed = time.time() - t0

    print(f"  Entrenamiento completado en {elapsed / 60:.1f} min")

    preds, labels, all_images, all_probs = evaluate_on_test(model, test_loader, device)

    metrics = compute_metrics(preds, labels)
    metrics["training_time_min"] = round(elapsed / 60, 2)

    print(f"  Macro F1 : {metrics['macro_f1']:.4f}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")

    plot_training_curves(
        history,
        save_path=os.path.join(out_dir, "training_curves.png"),
    )

    plot_confusion_matrix(
        preds,
        labels,
        save_path=os.path.join(out_dir, "confusion_matrix.png"),
    )

    plot_misclassified(
        all_images,
        all_probs,
        preds,
        labels,
        save_path=os.path.join(out_dir, "misclassified.png"),
        n=12,
    )

    save_metrics(metrics, os.path.join(out_dir, "metrics.json"))

    model_path = os.path.join(out_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"  ✓ Modelo guardado en {model_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Experimento 2 — Improvements")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Dataset a entrenar. 'all' ejecuta todos.",
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

    if all_metrics:
        print_summary_table(all_metrics)

        summary_path = os.path.join(RESULTS_DIR, "summary.json")

        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        print(f"  Resumen guardado en {summary_path}")


if __name__ == "__main__":
    main()