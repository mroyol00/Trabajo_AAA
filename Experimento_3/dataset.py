

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import IMG_SIZE, BATCH_SIZE, DATASETS, CLASSES, SEED


# ─────────────────────────────────────────────
# TRANSFORMACIONES
# ─────────────────────────────────────────────

# Train: augmentación ligera para generalización
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),   
    transforms.RandomRotation(15),            
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_dataloaders(dataset_key: str):
    root = DATASETS[dataset_key]

    # Verificar que existan las carpetas
    for split in ("train", "val", "test"):
        split_path = os.path.join(root, split)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(
                f"No se encontró la carpeta '{split}' en: {root}\n"
                f"Comprueba BASE_DATA_DIR en config.py"
            )

    train_ds = datasets.ImageFolder(
        root=os.path.join(root, "train"),
        transform=TRAIN_TRANSFORMS,
    )
    val_ds = datasets.ImageFolder(
        root=os.path.join(root, "val"),
        transform=EVAL_TRANSFORMS,
    )
    test_ds = datasets.ImageFolder(
        root=os.path.join(root, "test"),
        transform=EVAL_TRANSFORMS,
    )

    # Verificar que las clases coincidan con las esperadas
    detected = sorted(train_ds.classes)
    expected = sorted(CLASSES)
    if detected != expected:
        print(f"  ⚠ Clases detectadas {detected} ≠ esperadas {expected}")

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    print(f"  [{dataset_key}] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    return train_loader, val_loader, test_loader, train_ds.classes


if __name__ == "__main__":
    # Prueba rápida con el primer dataset disponible
    for key in DATASETS:
        try:
            tl, vl, tel, cls = get_dataloaders(key)
            imgs, labels = next(iter(tl))
            print(f"  Batch shape: {imgs.shape}, Labels: {labels[:4]}")
            break
        except FileNotFoundError as e:
            print(e)
