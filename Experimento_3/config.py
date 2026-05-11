

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
# Fíjate en la 'r' antes de las comillas
BASE_DATA_DIR = r"C:\Users\Usuario\Desktop\aaa\Trabajo_AAA\data"

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
LEARNING_RATE = 0.0003      # ANTES 0.001 -> Lo bajamos para que no tropiece
IMG_SIZE     = 224          
DROPOUT_P    = 0.3          
DROPOUT_FC_P = 0.4          
WEIGHT_DECAY = 1e-5         # ANTES 5e-5 -> Lo dejamos casi al mínimo

# ─────────────────────────────────────────────
# SALIDA
# ─────────────────────────────────────────────
RESULTS_DIR = "./results/exp3"

# Reproducibilidad
SEED = 42
