"""
config.py — Configuración central del Experimento 1 (Baseline CNN)
Ajusta BASE_DATA_DIR a la ruta raíz donde están tus 5 datasets.
"""

import os

# ─────────────────────────────────────────────
# RUTA RAÍZ DE LOS DATOS
# Cambia esto a la carpeta donde tienes los 5 datasets.
# Estructura esperada:
#   BASE_DATA_DIR/
#       dataset_grupo/      ← dataset del grupo
#           train/ val/ test/  (cada uno con subcarpetas por clase)
#       dataset_ODD/
#       dataset_EVEN/
#       dataset_COMBINED/
#       dataset_PUBLIC_split/
# ─────────────────────────────────────────────
BASE_DATA_DIR = "../data"   # ← CAMBIA ESTO

# Nombres de los 5 datasets y sus carpetas
DATASETS = {
    "group":    os.path.join(BASE_DATA_DIR, "group_split"),
    "odd":      os.path.join(BASE_DATA_DIR, "dataset_ODD","Kitchenware_ODD_pooled_Variable"),
    "even":     os.path.join(BASE_DATA_DIR, "dataset_EVEN","Kitchenware_EVEN_pooled_Homogeneous"),
    "combined": os.path.join(BASE_DATA_DIR, "dataset_COMBINED"),
    "public":   os.path.join(BASE_DATA_DIR, "dataset_PUBLIC_split"),
}

# Clases (orden fijo para matrices de confusión y métricas)
CLASSES = ["cups", "forks", "glasses", "knives", "plates", "spoons"]
NUM_CLASSES = len(CLASSES)

# ─────────────────────────────────────────────
# HIPERPARÁMETROS DE ENTRENAMIENTO
# ─────────────────────────────────────────────
BATCH_SIZE   = 16
EPOCHS       = 20
LEARNING_RATE = 0.001
IMG_SIZE     = 224          # imágenes 224×224
DROPOUT_P    = 0.25         # dropout en capas conv
DROPOUT_FC_P = 0.5          # dropout en capa fully-connected

# ─────────────────────────────────────────────
# SALIDA
# ─────────────────────────────────────────────
RESULTS_DIR = "./results/exp1"

# Reproducibilidad
SEED = 42
