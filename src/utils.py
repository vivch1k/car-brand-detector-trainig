import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns


def set_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    
    
def plot_curves(results: dict):
    range_epochs = list(range(len(results["train_loss"])))
      
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.lineplot(x=range_epochs, y=results["train_loss"], label="train_loss", color="red")
    sns.lineplot(x=range_epochs, y=results["test_loss"], label="test_loss", color="blue")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    sns.lineplot(x=range_epochs, y=results["train_acc"], label="train_acc", color="red")
    sns.lineplot(x=range_epochs, y=results["test_acc"], label="test_acc", color="blue")
    plt.grid(True)
    
    plt.savefig("data/artifacts/training_curves.png")