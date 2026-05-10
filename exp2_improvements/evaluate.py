import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

from config import CLASSES

def evaluate_on_test(model, test_loader, device):
    model.eval()
    all_preds, all_labels, all_images, all_probs = [], [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            probs   = torch.softmax(outputs, dim=1).cpu()
            preds   = outputs.argmax(dim=1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_images.append(images.cpu())
            all_probs.append(probs)

    all_images = torch.cat(all_images, dim=0)
    all_probs  = torch.cat(all_probs, dim=0)
    return all_preds, all_labels, all_images, all_probs


def compute_metrics(preds, labels):
    """Calcula y devuelve un dict con todas las métricas."""
    macro_f1  = f1_score(labels, preds, average="macro", zero_division=0)
    accuracy  = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall    = recall_score(labels, preds, average="macro", zero_division=0)

    report = classification_report(
        labels, preds,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "macro_f1":  round(macro_f1,  4),
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"],    4),
                "f1-score":  round(report[cls]["f1-score"],  4),
                "support":   int(report[cls]["support"]),
            }
            for cls in CLASSES
        },
    }
    return metrics

def plot_training_curves(history, save_path):
    """Guarda las curvas de loss y accuracy del entrenamiento."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # — Loss —
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   marker="o", markersize=3)
    axes[0].set_title("Loss por época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # — Accuracy —
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o", markersize=3)
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   marker="o", markersize=3)
    axes[1].set_title("Accuracy por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Curvas guardadas en {save_path}")


def plot_confusion_matrix(preds, labels, save_path):
    """Guarda la matriz de confusión normalizada y sin normalizar."""
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, fmt in [
        (axes[0], cm,      "Matriz de confusión (conteos)", "d"),
        (axes[1], cm_norm, "Matriz de confusión (normalizada)", ".2f"),
    ]:
        im = ax.imshow(data, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right")
        ax.set_yticklabels(CLASSES)

        thresh = data.max() / 2.0
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                val = f"{data[i, j]:{fmt}}"
                ax.text(j, i, val, ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black",
                        fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Matriz de confusión guardada en {save_path}")


def plot_misclassified(all_images, all_probs, preds, labels, save_path, n=12):
    """
    Guarda una cuadrícula con los primeros `n` ejemplos mal clasificados.
    Muestra imagen, clase real y clase predicha.
    """
    # Media y std de normalización (para desnormalizar visualmente)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    wrong_idx = [i for i, (p, l) in enumerate(zip(preds, labels)) if p != l]
    if not wrong_idx:
        print("  ℹ No hay ejemplos mal clasificados en test.")
        return

    wrong_idx = wrong_idx[:n]
    cols = 4
    rows = (len(wrong_idx) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for ax_i, img_i in enumerate(wrong_idx):
        img = all_images[img_i] * std + mean           # desnormalizar
        img = img.permute(1, 2, 0).clamp(0, 1).numpy()
        conf = all_probs[img_i][preds[img_i]].item()

        axes[ax_i].imshow(img)
        axes[ax_i].set_title(
            f"Real: {CLASSES[labels[img_i]]}\n"
            f"Pred: {CLASSES[preds[img_i]]} ({conf:.0%})",
            fontsize=7, color="red",
        )
        axes[ax_i].axis("off")

    for ax in axes[len(wrong_idx):]:
        ax.axis("off")

    plt.suptitle("Ejemplos mal clasificados", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Ejemplos mal clasificados guardados en {save_path}")