import torch
from data_loader import download_data, prepare_dataloaders, SimpleSegmentationDataset
from model import UNet, VGGUNet
from train import train_model
from visualize import visualize

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

batch_size = 4 # размер батча
num_classes = 35


def train(model, num_epochs, resize_size):
    train_dataset = download_data(r"./data/Resized images/Images/Train", resize_size)
    y_train = download_data(r"./data/Resized images/Masks/Train", resize_size)
    test_dataset = download_data(r"./data/Resized images/Images/Test", resize_size)
    y_test = download_data(r"./data/Resized images/Masks/Test", resize_size)

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

def visualize_test(model, resize_size):
    test_dataset = download_data(r"./data/Resized images/Images/Test", resize_size)
    y_test = download_data(r"./data/Resized images/Masks/Test", resize_size)
    dataset = SimpleSegmentationDataset(test_dataset, y_test)
    visualize(model, dataset)

#model = train(UNet(num_classes=num_classes), 25, (480, 256))
#torch.save(model.state_dict(), f"unet_epoch_25.pth") # можно включить сохранение модели на диск
#model = UNet(num_classes=num_classes)
#torch.load("unet_epoch_25.pth", model.state_dict())

#model = train(VGGUNet(num_classes=num_classes), 5, (484, 260))
#torch.save(model.state_dict(), f"vgg_epoch_25_2.pth")
model = VGGUNet(num_classes=num_classes)
torch.load("vgg_epoch_25_2.pth", model.state_dict())
visualize_test(model, (484, 260))
