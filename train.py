import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import my_model
import my_dataset

BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),
])


train_dataset =my_dataset.YouTubeVOSDataset(root_dir = "mini_dataset/valid", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.CrossEntropyLoss()

model = my_model.UNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()  
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        labels = labels.squeeze(1)
        optimizer.zero_grad()  # Обнуляем градиенты
        outputs = model(images)  # Прямой проход
        outputs = outputs.squeeze(1)
        #print("Outputs shape:", outputs.shape)  
        #print("Labels shape:", labels.shape) 
        loss = criterion(outputs, labels)  # Вычисляем лосс
        loss.backward()  # Обратное распространение ошибки
        optimizer.step()  # Обновляем веса
        
        running_loss += loss.item()
    
    # Выводим средний loss за эпоху
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}")

print("Обучение завершено!")

torch.save(model.state_dict(), "unet.pth")
print("Model saved.")

