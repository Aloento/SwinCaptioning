import torch
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from torchvision.transforms.functional import to_pil_image

from Model import Model
from persist import load_checkpoint
from prepare import prepare


def hot():
    idx_to_word, _, _, test_loader = prepare()

    model = Model(len(idx_to_word))

    epoch = load_checkpoint(model)
    model.eval()

    image, caption, indices = next(iter(test_loader))

    with torch.no_grad():
        output, weight = model(image, indices)

    predicted = torch.argmax(output, dim=2)
    predicted = predicted.squeeze(0).cpu().detach().numpy()
    predicted = [idx_to_word[idx] for idx in predicted]

    weight = weight[0]
    img_pil = to_pil_image(image[0])

    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15, 12))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        reshaped_weights = weight[i].reshape(16, 16).cpu().detach().numpy()

        min_weight = reshaped_weights.min()
        max_weight = reshaped_weights.max()
        normalized_weights = (reshaped_weights - min_weight) / (max_weight - min_weight)
        normalized_weights = 1 - zoom(normalized_weights, 16, order=0)

        ax.imshow(img_pil)
        ax.imshow(normalized_weights, cmap='gray', alpha=normalized_weights, vmin=0, vmax=1)

        ax.set_title(predicted[i])
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    hot()
