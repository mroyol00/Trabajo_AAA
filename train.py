import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR

from config import (
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    OPTIMIZER_NAME,
    MOMENTUM,
    LOSS_NAME,
    LABEL_SMOOTHING,
    FOCAL_GAMMA,
    USE_SCHEDULER,
    SCHEDULER_STEP_SIZE,
    SCHEDULER_GAMMA,
)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, outputs, targets):
        ce_loss = F.cross_entropy(outputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()


def get_loss_function():
    if LOSS_NAME == "cross_entropy":
        return nn.CrossEntropyLoss()

    if LOSS_NAME == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    if LOSS_NAME == "focal_loss":
        return FocalLoss(gamma=FOCAL_GAMMA)

    raise ValueError(f"Loss no reconocida: {LOSS_NAME}")


def get_optimizer(model):
    if OPTIMIZER_NAME == "adam":
        return Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

    if OPTIMIZER_NAME == "adamw":
        return AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

    if OPTIMIZER_NAME == "sgd_momentum":
        return SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )

    if OPTIMIZER_NAME == "rmsprop":
        return RMSprop(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )

    raise ValueError(f"Optimizador no reconocido: {OPTIMIZER_NAME}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return running_loss / total, correct / total, all_preds, all_labels


def train_model(model, train_loader, val_loader, device):
    criterion = get_loss_function()
    optimizer = get_optimizer(model)
    scheduler = None
    if USE_SCHEDULER:

        scheduler = StepLR(
            optimizer,
            step_size=SCHEDULER_STEP_SIZE,
            gamma=SCHEDULER_GAMMA
        )
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        vl_loss, vl_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

        print(
            f"Época {epoch:>2}/{EPOCHS} | "
            f"train_loss: {tr_loss:.4f} | train_acc: {tr_acc:.4f} | "
            f"val_loss: {vl_loss:.4f} | val_acc: {vl_acc:.4f}" 
        )
        if scheduler is not None:
            scheduler.step()
            
    if best_state is not None:
        model.load_state_dict(best_state)

    return history, val_preds, val_labels