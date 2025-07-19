import random

import torch
from matplotlib import pyplot as plt


def visualize(model, dataset):
    num_images = 6
    random_indices = random.sample(range(len(dataset)), num_images)

    fig, axes = plt.subplots(3, num_images, figsize=(20, 7))
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(random_indices):
            image, true_mask = dataset[idx]
            image_tensor = image.unsqueeze(0)

            output = model(image_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            axes[0, i].imshow(image.permute(1, 2, 0).cpu().numpy())
            axes[0, i].set_title(f"Изображение {idx}")
            axes[0, i].axis('off')

            axes[1, i].imshow(true_mask.squeeze().cpu().numpy())
            axes[1, i].set_title(f"Маска {idx}")
            axes[1, i].axis('off')

            axes[2, i].imshow(pred_mask)
            axes[2, i].set_title(f"Результат {idx}")
            axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()