import os

BASE_DATA_DIR = "../data"

DATASETS = {
    "group":    os.path.join(BASE_DATA_DIR, "group_split"),
    "odd":      os.path.join(BASE_DATA_DIR, "dataset_ODD", "Kitchenware_ODD_pooled_Variable"),
    "even":     os.path.join(BASE_DATA_DIR, "dataset_EVEN", "Kitchenware_EVEN_pooled_Homogeneous"),
    "combined": os.path.join(BASE_DATA_DIR, "dataset_COMBINED"),
    "public":   os.path.join(BASE_DATA_DIR, "dataset_PUBLIC_split"),
}

CLASSES = ["cups", "forks", "glasses", "knives", "plates", "spoons"]
NUM_CLASSES = len(CLASSES)

BATCH_SIZE = 16
EPOCHS = 20
IMG_SIZE = 224
SEED = 42

# =========================
# MODO DE EXPERIMENTO
# =========================
EXPERIMENT_NAME = "exp2_regularization"
# opciones:
# "exp2_regularization"
# "exp2_optimizer"
# "exp2_loss"
# "exp2_best_combination"

# =========================
# REGULARIZACIÓN
# =========================
USE_AUGMENTATION = True

USE_CONV_DROPOUT = False

DROPOUT_CONV_P = 0.0
DROPOUT_FC_P = 0.3
WEIGHT_DECAY = 1e-4

# =========================
# OPTIMIZACIÓN
# =========================
OPTIMIZER_NAME = "sgd_momentum"
# opciones: "adam", "adamw", "sgd_momentum", "rmsprop"

LEARNING_RATE = 0.001
MOMENTUM = 0.9

# =========================
# SCHEDULER
# =========================
USE_SCHEDULER = True
SCHEDULER_STEP_SIZE = 7
SCHEDULER_GAMMA = 0.5

# =========================
# FUNCIÓN DE PÉRDIDA
# =========================
LOSS_NAME = "cross_entropy"# opciones: "cross_entropy", "label_smoothing", "focal_loss"

LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0

RUN_NAME = (
    f"{EXPERIMENT_NAME}"
    f"_opt-{OPTIMIZER_NAME}"
    f"_loss-{LOSS_NAME}"
    f"_aug-{USE_AUGMENTATION}"
    f"_convdrop-{USE_CONV_DROPOUT}"
    f"_wd-{WEIGHT_DECAY}"
    f"_lr-{LEARNING_RATE}"
)

RESULTS_DIR = f"./results/{RUN_NAME}"