import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models



class SimpleSegmentationDataset():
  def __init__(self, image, mask):
    self.data = image
    self.y = mask
    self.transform = transforms.Compose(
        [transforms.ToTensor() ] # <-- это приводит HWC → CHW, т.е меняет местами порядок каналов в тензоре изображения
    )
    # стандартный порядок: ширина, высота, цветовые каналы
    # но пайторч требует: цветовые каналы, ширина, высота

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    image = self.data[idx]
    image = self.transform(image)
    target = self.y[idx]
    target = self.transform(target)
    return image, target


def download_data(path):
    data = []
    for path_image in sorted(os.listdir(path=path)):
        image = Image.open(path + "/" + path_image)
        data.append(np.array(image))
    return data

def prepare_dataloaders(train_dataset, y_train, test_dataset, y_test, batch_size):
    train_dataset = SimpleSegmentationDataset(train_dataset, y_train)
    test_dataset = SimpleSegmentationDataset(test_dataset, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


