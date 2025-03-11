import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from my_model import UNet 
from my_dataset import YouTubeVOSDataset  


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 5

def load_model(model_path):
    model = UNet().to(DEVICE)  
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))  
    model.eval()  
    return model

# 2. Подготовка данных
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

test_dataset = YouTubeVOSDataset(root_dir="mini_dataset/valid", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = load_model("unet.pth")

def plot_predictions(model, loader, device, num_examples=5):
    model.eval()
    with torch.no_grad():
        images, true_masks = next(iter(loader))
        images = images.to(device)
        true_masks = true_masks.to(device)
        
        preds = torch.sigmoid(model(images)) 
        preds = (preds > 0.5).float() 
          
        images = images.cpu().numpy()
        true_masks = true_masks.cpu().numpy()
        preds = preds.cpu().numpy()
        
        fig, ax = plt.subplots(num_examples, 3, figsize=(12, 4*num_examples))
        
        for i in range(num_examples):
            img = images[i].transpose(1, 2, 0)
            
            true_mask = true_masks[i].squeeze()
            
            pred_mask = preds[i].squeeze()
            
            ax[i, 0].imshow(img)
            ax[i, 0].set_title("Input Image")
            ax[i, 0].axis("off")
            
            ax[i, 1].imshow(true_mask, cmap="gray")
            ax[i, 1].set_title("True Mask")
            ax[i, 1].axis("off")
            
            ax[i, 2].imshow(pred_mask, cmap="gray")
            ax[i, 2].set_title("Predicted Mask")
            ax[i, 2].axis("off")
        
        plt.tight_layout()
        plt.show()

plot_predictions(model, test_loader, DEVICE, num_examples=5)