import os
from collections import Counter

import pandas as pd
import requests
import zipfile

from torch.utils.data import DataLoader

from FlickrDataset import FlickrDataset


def download_and_extract(url, download_path, extract_path):
    if os.path.exists(download_path):
        print(f"{download_path} already exists")
        return

    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(download_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {download_path}")
    else:
        print(f"Failed to download from {url}")

    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    print(f"Extracted {download_path}")


def prepare_data():
    base_dir = 'Flickr8k'
    os.makedirs(base_dir, exist_ok=True)

    text_zip_path = os.path.join(base_dir, base_dir + '_text.zip')
    dataset_zip_path = os.path.join(base_dir, base_dir + '_Dataset.zip')

    url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_"

    download_and_extract(url + 'text.zip', text_zip_path, base_dir)

    download_and_extract(url + 'Dataset.zip', dataset_zip_path, base_dir)


def prepare_des():
    description_file = os.path.join('Flickr8k', 'Flickr8k.token.txt')
    descriptions = pd.read_csv(description_file, sep='\t', header=None, names=['image', 'caption'])
    descriptions["image"] = descriptions["image"].apply(lambda x: x.split('#')[0])

    return descriptions


def build_vocab(descriptions):
    counter = Counter()

    for caption in descriptions['caption']:
        tokens = caption.lower().split()
        counter.update(tokens)

    words = [word for word, count in counter.items() if count >= 2]
    # words = [word for word, count in counter.items()]
    words = ['[PAD]', '[EOF]', '[UNK]'] + sorted(words)

    word_to_idx = {word: idx for idx, word in enumerate(words)}
    return word_to_idx, words


def caption_to_indices(caption, word_to_index):
    caption = caption.lower().split()
    caption = caption + ['[EOF]']
    caption = [word_to_index.get(word, word_to_index['[UNK]']) for word in caption]

    caption = caption[:20]
    caption = caption + [word_to_index['[PAD]']] * (20 - len(caption))

    return caption


def prepare_loader(des, vocab, eval=False):
    train_dataset = FlickrDataset(des, vocab, "train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=100,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    )

    val_dataset = FlickrDataset(des, vocab, "dev")
    val_loader = DataLoader(
        val_dataset,
        batch_size=50,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    test_dataset = FlickrDataset(des, vocab, "test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=100 if eval else 1,
        shuffle=True
    )

    return train_loader, val_loader, test_loader


def prepare(eval=False) -> tuple[list[str], DataLoader, DataLoader, DataLoader]:
    prepare_data()

    des = prepare_des()
    word_to_idx, idx_to_word = build_vocab(des)

    des['indices'] = des['caption'].apply(lambda x: caption_to_indices(x, word_to_idx))

    train_loader, val_loader, test_loader = prepare_loader(des, word_to_idx, eval)

    return idx_to_word, train_loader, val_loader, test_loader


if __name__ == '__main__':
    prepare()
