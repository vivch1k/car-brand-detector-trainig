import torch
import numpy as np
import torch.nn as nn
import torchmetrics
from collections import defaultdict


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        # Warm-up
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * num_cycles * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_steps(model: nn.Module, 
                train_dataloader: torch.utils.data.DataLoader, 
                eval_metric: torchmetrics.Metric, 
                optimizer: torch.optim.Optimizer, 
                device: torch.device, 
                scheduler):
    
    train_loss, train_acc = 0, 0
    
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        outputs = model(**batch)
        
        loss = outputs.loss
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        probs = torch.softmax(outputs.logits, dim=1)
        label = torch.argmax(probs, dim=1)
        acc = eval_metric(label, batch["labels"])
        train_acc += acc.item()
    
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    
    return train_loss, train_acc


def test_steps(model: nn.Module,
               test_dataloader: torch.utils.data.DataLoader,
               eval_metric: torchmetrics.Metric,
               device: torch.device):
    
    test_loss, test_acc = 0, 0
    
    with torch.inference_mode():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
        
            loss = outputs.loss
            test_loss += loss.item()
            
            probs = torch.softmax(outputs.logits, dim=1)
            label = torch.argmax(probs, dim=1)
            acc = eval_metric(label, batch["labels"])
            test_acc += acc.item()
            
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    
    return test_loss, test_acc


def train(model: nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          eval_metric: torchmetrics.Metric,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          warmup_steps: int,
          device: torch.device):
    
    results = defaultdict(list)
    
    total_steps = epochs * len(train_dataloader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    for epoch in range(epochs):        
        train_loss, train_acc = train_steps(model=model,
                                            train_dataloader=train_dataloader,
                                            eval_metric=eval_metric,
                                            optimizer=optimizer,
                                            device=device,
                                            scheduler=scheduler)
        
        test_loss, test_acc = test_steps(model=model,
                                         test_dataloader=test_dataloader,
                                         eval_metric=eval_metric,
                                         device=device)
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | ",
              f"Trian loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f} |",
              f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
        
        checkpoint = {
            "epoch": epoch+1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        torch.save(checkpoint, f"models/fine-tuning/vit_car_{epoch+1}e_unfreeze-4.pth")       
        
    return results