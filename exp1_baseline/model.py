"""
model.py — BaselineCNN (Experimento 1, sin modificaciones)

Arquitectura exacta del enunciado:
  Input RGB 224×224
  Conv 3→32 + BN + ReLU + MaxPool + Dropout
  Conv 32→64 + BN + ReLU + MaxPool + Dropout
  Conv 64→128 + BN + ReLU + MaxPool + Dropout
  Flatten
  Linear 128*28*28 → 256 + ReLU + Dropout
  Linear 256 → 6
"""

import torch
import torch.nn as nn
from config import NUM_CLASSES, DROPOUT_P, DROPOUT_FC_P


class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()

        # ── Bloque 1: Conv 3 → 32 ──────────────────────────────
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 224 → 112
            nn.Dropout2d(p=DROPOUT_P),
        )

        # ── Bloque 2: Conv 32 → 64 ─────────────────────────────
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 112 → 56
            nn.Dropout2d(p=DROPOUT_P),
        )

        # ── Bloque 3: Conv 64 → 128 ────────────────────────────
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 56 → 28
            nn.Dropout2d(p=DROPOUT_P),
        )

        # ── Clasificador fully-connected ───────────────────────
        # Tras 3 MaxPool(2) sobre 224: 224/2/2/2 = 28
        # Feature map: 128 × 28 × 28 = 100352
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT_FC_P),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


def get_model(device):
    """Instancia el modelo y lo mueve al device."""
    model = BaselineCNN().to(device)
    return model


if __name__ == "__main__":
    # Verificación rápida de dimensiones
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    dummy = torch.zeros(1, 3, 224, 224).to(device)
    out = model(dummy)
    print(f"Modelo OK — Salida: {out.shape}")   # → torch.Size([1, 6])
    print(model)
