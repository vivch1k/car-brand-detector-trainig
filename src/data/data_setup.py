import torch
import torchvision
from torch.utils.data import DataLoader 
from torchvision import datasets
import transformers
from typing import Optional, Callable

def create_dataloaders(train_path: str=None,
                       test_path: str=None,
                       train_transform: torchvision.transforms=None,
                       test_transform: torchvision.transforms=None,
                       batch_size: int=32,
                       num_worker: int=0,
                       pin_memory: bool=False,
                       processor: transformers.AutoImageProcessor=None):
    
    def collate_fn(batch):
        img, label = zip(*batch)
        encodings = processor(list(img), return_tensors="pt")
        encodings["labels"] = torch.tensor(label)    
        return encodings
    
    train_data, train_dataloader = None, None
    test_data, test_dataloader = None, None
    collate_fn_to_use = collate_fn if processor is not None else None
    
    if train_path is not None and train_transform is not None:
        train_data = datasets.ImageFolder(root=train_path,
                                        transform=train_transform)    
        train_dataloader = DataLoader(dataset=train_data,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_worker,
                                    pin_memory=pin_memory,
                                    collate_fn=collate_fn_to_use)
    
    if test_path is not None and test_transform is not None:
        test_data = datasets.ImageFolder(root=test_path,
                                        transform=test_transform)
        test_dataloader = DataLoader(dataset=test_data,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_worker,
                                    pin_memory=pin_memory,
                                    collate_fn=collate_fn_to_use)
    
    class_names = train_data.classes if train_data is not None else test_data.classes
    
    return train_data, test_data, train_dataloader, test_dataloader, class_names