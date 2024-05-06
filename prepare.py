import os

import pandas as pd
import requests
import zipfile


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
    vocab = set()
    vocab.update(['<start>', '<end>', '<unk>'])

    for caption in descriptions['caption']:
        vocab.update(caption.lower().split())

    l_vocab = sorted(list(vocab))

    word_to_idx = {word: idx for idx, word in enumerate(l_vocab)}
    return word_to_idx


def caption_to_indices(caption, word_to_index):
    caption = caption.lower().split()
    caption = ['<start>'] + caption + ['<end>']
    caption = [word_to_index.get(word, word_to_index['<unk>']) for word in caption]

    return caption


def prepare():
    prepare_data()
    des = prepare_des()
    vocab = build_vocab(des)

    des['indices'] = des['caption'].apply(lambda x: caption_to_indices(x, vocab))

    return des, vocab


if __name__ == '__main__':
    prepare()
