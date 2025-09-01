import torch
from transformers import (AutoImageProcessor,
                          AutoModelForImageClassification)
from torchvision import transforms
from torchmetrics import Accuracy
import numpy as np

from models.engine import train
from data.data_setup import create_dataloaders
from utils import set_seeds


train_path = "data/train"
val_path = "data/val"
BATCH_SIZE = 64
SEED = 42
EPOCHS = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    set_seeds(SEED)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])


    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    
    processor = AutoImageProcessor.from_pretrained(
        "therealcyberlord/stanford-car-vit-patch16",
        use_fast=True
    )
    
    train_dataloader, val_dataloader, class_names = create_dataloaders(train_path=train_path,
                                                                       test_path=val_path,
                                                                       train_transform=train_transform,
                                                                       test_transform=val_transform,
                                                                       batch_size=BATCH_SIZE,
                                                                       processor=processor)
    
    
    model = AutoModelForImageClassification.from_pretrained(
        "therealcyberlord/stanford-car-vit-patch16",
        num_labels=len(class_names),
        ignore_mismatched_sizes=True
    ).to(device)
    
    checkpoint = torch.load("models/vit_car_checkpoint_50e_unfreeze-2.pth")
    model.load_state_dict(checkpoint["model_state"])
    
    for param in model.vit.parameters():
        param.requires_grad = False
    
    for param in model.vit.encoder.layer[-4:].parameters():
        param.requires_grad = True
    
    accuracy = Accuracy(task="multiclass", num_classes=len(class_names)).to(device)
    optim = torch.optim.AdamW([
        {"params": model.vit.encoder.layer[-4:].parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 3e-5}
    ], weight_decay=0.05)
      
    
    results = train(model=model,
                    train_dataloader=train_dataloader,
                    test_dataloader=val_dataloader,
                    eval_metric=accuracy,
                    optimizer=optim,
                    epochs=EPOCHS,
                    warmup_steps=3*len(train_dataloader),
                    device=device)
    
    np.save("outputs/training_results/fine-tuning_results_vit_car_50e_unfreeze-4.npy", results)