import torch
import torch.nn as nn
from torchvision import models
from config import DROPOUT_FC_P

# Añadimos un valor por defecto num_classes=6 por si acaso
def get_model(device=None, num_classes=6):
    # 1. Cargamos MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # 2. Congelamos parámetros
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. Ajustamos la salida
    in_features = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_FC_P),
        nn.Linear(in_features, num_classes) 
    )
    
    # Movemos al device (cpu)
    if device:
        model = model.to(device)
        
    return model