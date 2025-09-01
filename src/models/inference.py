import torch
from torchvision import transforms
from PIL import Image


def img_prediction(img_path,
                   model,
                   processor):
    
    image = Image.open(img_path)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    
    input_image = processor(transform(image), return_tensors="pt")
    
    model.eval()
    with torch.inference_mode():
        output = model(**input_image)
        prob = torch.max(torch.softmax(output.logits, 1), 1)[0]
        label = torch.argmax(output.logits, 1)
    
    
    return prob.item(), label.item()