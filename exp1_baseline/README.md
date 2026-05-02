# Experimento 1 — Baseline CNN

Clasificación de cubertería en 6 clases: `cups, forks, glasses, knives, plates, spoons`.

## Estructura del proyecto

```
exp1_baseline/
├── config.py       ← Rutas e hiperparámetros  ← EDITA ESTE PRIMERO
├── dataset.py      ← DataLoaders (ImageFolder)
├── model.py        ← BaselineCNN (arquitectura exacta del enunciado)
├── train.py        ← Bucle entrenamiento / validación
├── evaluate.py     ← Métricas, figuras, ejemplos mal clasificados
├── utils.py        ← Semillas, device, tabla resumen
└── main.py         ← Punto de entrada principal
```

## Resultado esperado

```
results/exp1/
├── group/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── misclassified.png
│   ├── metrics.json
│   └── model.pth
├── odd/    (ídem)
├── even/   (ídem)
├── combined/ (ídem)
├── public/ (ídem)
└── summary.json    ← tabla resumen con Macro F1 y Accuracy
```

---

## Uso en VS Code (local)

### 1. Instalar dependencias

```bash
pip install torch torchvision matplotlib scikit-learn
```

> Con GPU NVIDIA instala la versión CUDA:
> https://pytorch.org/get-started/locally/

### 2. Ajustar rutas en `config.py`

```python
BASE_DATA_DIR = "./data"   # carpeta que contiene las 5 subcarpetas de datasets
```

Estructura esperada:
```
data/
├── dataset_grupo/
│   ├── train/cups/ train/forks/ ... train/spoons/
│   ├── val/  ...
│   └── test/ ...
├── dataset_ODD/
├── dataset_EVEN/
├── dataset_COMBINED/
└── dataset_PUBLIC_split/
```

### 3. Ejecutar

```bash
# Todos los datasets (recomendado)
python main.py

# Solo uno (para probar rápido)
python main.py --dataset group
python main.py --dataset odd
python main.py --dataset even
python main.py --dataset combined
python main.py --dataset public
```

---

## Uso en Google Colab

Crea un notebook y copia estas celdas:

### Celda 1 — Subir archivos y montar Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Opción A: datos en Drive
BASE_DATA_DIR = "/content/drive/MyDrive/datos_practica"

# Opción B: subir zip
# from google.colab import files
# files.upload()   # sube exp1_baseline.zip y los datos
# !unzip exp1_baseline.zip
# !unzip datos.zip -d data/
```

### Celda 2 — Instalar dependencias

```python
!pip install -q scikit-learn
import torch
print(torch.cuda.is_available())   # debe ser True en Colab con GPU
```

### Celda 3 — Ajustar config y ejecutar

```python
import subprocess
# Edita BASE_DATA_DIR en config.py si es necesario
subprocess.run(["python", "main.py"], check=True)
```

O ejecuta directamente desde el notebook importando `main`:

```python
import sys
sys.path.insert(0, "/content/exp1_baseline")

# Sobrescribir ruta si es necesario
import config
config.BASE_DATA_DIR = "/content/drive/MyDrive/datos_practica"
config.DATASETS = {
    "group":    config.BASE_DATA_DIR + "/dataset_grupo",
    "odd":      config.BASE_DATA_DIR + "/dataset_ODD",
    "even":     config.BASE_DATA_DIR + "/dataset_EVEN",
    "combined": config.BASE_DATA_DIR + "/dataset_COMBINED",
    "public":   config.BASE_DATA_DIR + "/dataset_PUBLIC_split",
}

from main import main
main()
```

---

## Hiperparámetros (enunciado)

| Parámetro     | Valor     |
|---------------|-----------|
| Optimizer     | Adam      |
| Learning rate | 0.001     |
| Loss          | CrossEntropyLoss |
| Epochs        | 20        |
| Batch size    | 16        |
| Input size    | 224×224   |

## Tabla de resultados (rellenar tras ejecutar)

| Experimento | Group | ODD | EVEN | ODD+EVEN | Public |
|-------------|-------|-----|------|----------|--------|
| Exp 1 (Macro F1) | — | — | — | — | — |
| Exp 1 (Accuracy) | — | — | — | — | — |

Los valores exactos estarán en `results/exp1/summary.json` tras ejecutar.
