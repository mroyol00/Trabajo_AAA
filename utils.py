import os
import json
import random
import numpy as np
import torch
from config import SEED, DATASETS


# ─────────────────────────────────────────────
# REPRODUCIBILIDAD
# ─────────────────────────────────────────────

def set_seed(seed: int = SEED):
   
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def get_device():
    """Devuelve el device disponible (MPS > CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():   # Apple Silicon
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")
    return device

def save_metrics(metrics: dict, path: str):
    """Guarda el diccionario de métricas en un archivo JSON."""
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Métricas guardadas en {path}")


def load_metrics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def print_summary_table(all_metrics: dict):

    keys = list(DATASETS.keys())
    header = f"{'Experimento':<14}" + "".join(f"{k:>12}" for k in keys)
    sep    = "-" * (14 + 12 * len(keys))

    print("\n" + "=" * len(sep))
    print("  TABLA RESUMEN — Experimento 1 (Baseline CNN)")
    print("=" * len(sep))

    print(header)
    print(sep)

    for metric_name in ("macro_f1", "accuracy"):
        row = f"  {metric_name:<12}"
        for k in keys:
            val = all_metrics.get(k, {}).get(metric_name, "N/A")
            row += f"{val:>12}" if isinstance(val, str) else f"{val:>12.4f}"
        print(row)

    print(sep)
    print()


def make_output_dir(results_dir: str, dataset_key: str) -> str:
    """Crea y devuelve el directorio de salida para un dataset."""
    path = os.path.join(results_dir, dataset_key)
    os.makedirs(path, exist_ok=True)
    return path
