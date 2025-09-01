import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchmetrics import Accuracy, ConfusionMatrix

import matplotlib.pyplot as plt
import seaborn as sns

from data.data_setup import create_dataloaders
from models.engine import eval_model

test_path = "data/val"
BATCH_SIZE = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    
    processor = AutoImageProcessor.from_pretrained(
        "therealcyberlord/stanford-car-vit-patch16",
        use_fast=True
    )
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    
    _, test_data, _, test_dataloader, class_names = create_dataloaders(test_path=test_path,
                                                                       test_transform=test_transform,
                                                                       batch_size=BATCH_SIZE,
                                                                       processor=processor)
    
    model = AutoModelForImageClassification.from_pretrained(
        "therealcyberlord/stanford-car-vit-patch16",
        num_labels=len(class_names),
        ignore_mismatched_sizes=True
    ).to(device)
    
    checkpoint = torch.load("models/fine-tuning/vit_car_50e_unfreeze-4.pth")
    model.load_state_dict(checkpoint["model_state"])
    
    pred_labels, pred_probs = eval_model(model=model,
                                         test_dataloader=test_dataloader,
                                         device=device)
    true_labels = torch.tensor(test_data.targets, device=device)
    
    
    accuracy = Accuracy(task="multiclass", num_classes=len(class_names)).to(device)
    print(f"Accuracy: {accuracy(pred_labels, true_labels).item()}")
    
    confmat = ConfusionMatrix(task="multiclass",num_classes=len(class_names)).to(device)
    confmat_tensor = confmat(pred_labels, true_labels)
    confmat_numpy = confmat_tensor.cpu().numpy()
    
    plt.figure(figsize=(20, 20))
    sns.heatmap(confmat_numpy,
                annot=True,
                fmt=".0f",
                cbar=False,
                xticklabels=class_names,
                yticklabels=class_names)
    plt.savefig("model_results/conf_matrix.png")
            