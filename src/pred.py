import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pathlib import Path
import matplotlib.pyplot as plt
import os

from models.inference import img_prediction

data_test_path = Path("data/img_for_test")
img_name = "ford.jpg"
img_path = data_path / img_name
class_names = ['acura', 'aston martin', 'audi',
               'bentley', 'bmw', 'buick', 'byd',
               'cadillac', 'chevrolet', 'chrysler',
               'dodge', 'fiat', 'ford', 'gmc', 'honda',
               'hyundai', 'infiniti', 'jaguar', 'jeep',
               'kia', 'lada', 'land rover', 'lexus',
               'lincoln', 'mazda', 'mercedes-benz',
               'mini', 'mitsubishi', 'nissan',
               'porsche', 'ram', 'renault',
               'skoda', 'subaru', 'suzuki',
               'toyota', 'volkswagen', 'volvo']


if __name__ == "__main__":
    
    processor = AutoImageProcessor.from_pretrained(
        "therealcyberlord/stanford-car-vit-patch16",
        use_fast=True
    )
    
    model = AutoModelForImageClassification.from_pretrained(
        "therealcyberlord/stanford-car-vit-patch16",
        num_labels=38,
        ignore_mismatched_sizes=True
    )
    checkpoint = torch.load("models/fine-tuning/vit_car_50e_unfreeze-4.pth")
    model.load_state_dict(checkpoint["model_state"])
        
    prob, label = img_prediction(img_path,
                                 model,
                                 processor)
    print(f"Prob: {prob:.2f}",
          f"Label: {label}",
          f"Class: {class_names[label]}")
    
    