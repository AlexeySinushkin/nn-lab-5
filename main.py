import torch
import matplotlib.pyplot as plt
from data_loader import download_data, prepare_dataloaders, SimpleSegmentationDataset
from model import UNet
from train import train_model
from visualize import visualize

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

num_epochs = 25  # количество эпох
batch_size = 8 # размер батча
num_classes = 35


def train(model):
    train_dataset = download_data(r"./data/Resized images/Images/Train")
    y_train = download_data(r"./data/Resized images/Masks/Train")
    test_dataset = download_data(r"./data/Resized images/Images/Test")
    y_test = download_data(r"./data/Resized images/Masks/Test")

    # гипер-параметры обучения
    learning_rate = 0.0001 # скорость обучения

    train_loader, test_loader = prepare_dataloaders(train_dataset, y_train, test_dataset, y_test, batch_size)
    model, train_losses, val_losses = train_model(model,
                                             train_loader,
                                             test_loader,
                                             num_epochs=num_epochs,
                                             learning_rate=learning_rate)

    test_loss = val_losses[-1]
    print(f"\nРезультаты на тестовой выборке:\nПотери: {test_loss:.4f}")
    return model

def visualize_test(model):
    test_dataset = download_data(r"./data/Resized images/Images/Test")
    y_test = download_data(r"./data/Resized images/Masks/Test")
    dataset = SimpleSegmentationDataset(test_dataset, y_test)
    visualize(model, dataset)

#model = run(UNet(num_classes=num_classes))
#torch.save(model.state_dict(), f"unet_epoch_25.pth") # можно включить сохранение модели на диск
model = UNet(num_classes=num_classes)
torch.load("unet_epoch_25.pth", model.state_dict())
visualize_test(model)
