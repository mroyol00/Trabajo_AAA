"""
train.py — Bucle de entrenamiento y validación (Experimento 1)

Devuelve historial de métricas por época para poder graficar
las curvas de entrenamiento.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from config import LEARNING_RATE, EPOCHS


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Entrena el modelo durante una época y devuelve la pérdida media."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Evalúa el modelo en val/test. Devuelve loss, accuracy y predicciones."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += images.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


def train_model(model, train_loader, val_loader, device):
    """
    Entrena el modelo durante EPOCHS épocas.

    Returns
    -------
    history : dict con listas por época:
        'train_loss', 'val_loss', 'train_acc', 'val_acc'
    val_preds, val_labels : predicciones de la última época en val
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [],
        "val_loss":   [],
        "train_acc":  [],
        "val_acc":    [],
    }

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        # Guardar el mejor modelo (menor val_loss)
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"  Época {epoch:>2}/{EPOCHS} | "
            f"train_loss: {tr_loss:.4f} | train_acc: {tr_acc:.4f} | "
            f"val_loss: {vl_loss:.4f} | val_acc: {vl_acc:.4f}"
        )

    # Restaurar el mejor estado antes de devolver
    if best_state is not None:
        model.load_state_dict(best_state)

    return history, val_preds, val_labels
