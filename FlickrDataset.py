import os
import pandas as pd
import torch
from PIL import Image
from pandas import DataFrame
from torch import FloatTensor, LongTensor

from torch.utils.data import Dataset
from torchvision.transforms import transforms


class FlickrDataset(Dataset):
    def __init__(self, descriptions: DataFrame, word_to_index: dict[str, int], dataset_type='train'):
        self.dataset_type = dataset_type
        self.word_to_index = word_to_index

        self.images_dir = os.path.join('Flickr8k', 'Flicker8k_Dataset')

        images_list = os.path.join('Flickr8k', 'Flickr_8k.' + dataset_type + 'Images.txt')
        images = pd.read_csv(images_list, header=None, names=['image'])

        self.descriptions = descriptions[descriptions['image'].isin(images['image'])]

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx) -> tuple[FloatTensor, str, LongTensor]:
        caption = self.descriptions.iloc[idx]['caption']
        indices = self.descriptions.iloc[idx]['indices']

        img_path = os.path.join(self.images_dir, self.descriptions.iloc[idx]['image'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        indices = torch.tensor(indices).long()
        return image, caption, indices


if __name__ == '__main__':
    dev_list = pd.read_csv(os.path.join('Flickr8k', 'Flickr_8k.devImages.txt'), header=None, names=['image'])
    test_list = pd.read_csv(os.path.join('Flickr8k', 'Flickr_8k.testImages.txt'), header=None, names=['image'])
    train_list = pd.read_csv(os.path.join('Flickr8k', 'Flickr_8k.trainImages.txt'), header=None, names=['image'])

    dev_set = set(dev_list['image'])
    test_set = set(test_list['image'])
    train_set = set(train_list['image'])

    # 0
    print(len(dev_set & test_set))
    print(len(dev_set & train_set))
    print(len(test_set & train_set))
    print(len(dev_set & test_set & train_set))
    print(len(dev_set | test_set | train_set))
