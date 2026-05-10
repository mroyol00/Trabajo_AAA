import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import IMG_SIZE, BATCH_SIZE, DATASETS, CLASSES, SEED, USE_AUGMENTATION


if USE_AUGMENTATION:
    TRAIN_TRANSFORMS = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
else:
    TRAIN_TRANSFORMS = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
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

    for split in ("train", "val", "test"):
        split_path = os.path.join(root, split)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(f"No se encontró '{split}' en {root}")

    train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=TRAIN_TRANSFORMS)
    val_ds = datasets.ImageFolder(os.path.join(root, "val"), transform=EVAL_TRANSFORMS)
    test_ds = datasets.ImageFolder(os.path.join(root, "test"), transform=EVAL_TRANSFORMS)

    detected = sorted(train_ds.classes)
    expected = sorted(CLASSES)

    if detected != expected:
        print(f"⚠ Clases detectadas {detected} ≠ esperadas {expected}")

    g = torch.Generator()
    g.manual_seed(SEED)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=pin,
        generator=g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=pin,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=pin,
    )

    print(f"[{dataset_key}] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    return train_loader, val_loader, test_loader, train_ds.classes