import os
import pandas as pd
from PIL import Image
from pandas import DataFrame

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
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.descriptions.iloc[idx]['image'])
        caption = self.descriptions.iloc[idx]['caption']

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image, caption
