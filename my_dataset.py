import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class YouTubeVOSDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=(256, 256)):
        """
        root_dir: Папка с датасетом YouTube-VOS.
        transform: Аугментации и преобразования для изображений.
        """
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        # Пути к изображениям и маскам
        self.image_dir = os.path.join(root_dir, "JPEGImages")  
        self.mask_dir = os.path.join(root_dir, "Annotations")  

        self.video_folders = sorted(os.listdir(self.image_dir)) 

        # Собираем список всех кадров и соответствующих масок
        self.data = []
        for video in self.video_folders:
            video_path = os.path.join(self.image_dir, video)
            mask_path = os.path.join(self.mask_dir, video)

            if not os.path.exists(mask_path):
                continue  

            frames = sorted(os.listdir(video_path))
            for frame in frames:
                frame_path = os.path.join(video_path, frame)
                mask_frame_path = os.path.join(mask_path, frame.replace(".jpg", ".png")) 

                if os.path.exists(mask_frame_path):
                    self.data.append((frame_path, mask_frame_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path = self.data[idx]

        # Загружаем изображение и маску
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Применяем преобразования
        image = self.transform(image)
        mask = self.transform(mask) 

        return image, mask

dataset = YouTubeVOSDataset("mini_dataset/valid")


