import torchvision
from torch.utils.data import DataLoader 
from torchvision import datasets


def create_dataloaders(train_path: str,
                       test_path: str,
                       train_transform: torchvision.transforms,
                       test_transform: torchvision.transforms,
                       batch_size: int,
                       num_worker: int=0,
                       pin_memory: bool=False,
                       collate_fn: function=None):
    
    train_data = datasets.ImageFolder(root=train_path,
                                      transform=train_transform)
    
    test_data = datasets.ImageFolder(root=test_path,
                                    transform=test_transform)
    
    class_names = train_data.classes
    
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_worker,
                                  pin_memory=pin_memory,
                                  collate_fn=collate_fn)
    
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_worker,
                                 pin_memory=pin_memory,
                                 collate_fn=collate_fn)
    
    return train_dataloader, test_dataloader, class_names