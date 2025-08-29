import torch
import torchvision
from torch.utils.data import DataLoader 
from torchvision import datasets
from typing import Optional, Callable

def create_dataloaders(train_path: str,
                       test_path: str,
                       train_transform: torchvision.transforms,
                       test_transform: torchvision.transforms,
                       batch_size: int,
                       num_worker: int=0,
                       pin_memory: bool=False,
                       processor: Optional[Callable]=None):
    
    def collate_fn(batch):
        img, label = zip(*batch)
        encodings = processor(list(img), return_tensors="pt")
        encodings["labels"] = torch.tensor(label)
        
        return encodings
    
    train_data = datasets.ImageFolder(root=train_path,
                                      transform=train_transform)
    
    test_data = datasets.ImageFolder(root=test_path,
                                    transform=test_transform)
    
    class_names = train_data.classes
    
    collate_fn_to_use = collate_fn if processor is not None else None
    
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_worker,
                                  pin_memory=pin_memory,
                                  collate_fn=collate_fn_to_use)
    
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_worker,
                                 pin_memory=pin_memory,
                                 collate_fn=collate_fn_to_use)
    
    return train_dataloader, test_dataloader, class_names