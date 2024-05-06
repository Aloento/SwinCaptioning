import os
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


if __name__ == '__main__':
    base_dir = 'Flickr8k'
    os.makedirs(base_dir, exist_ok=True)

    text_zip_path = os.path.join(base_dir, base_dir + '_text.zip')
    dataset_zip_path = os.path.join(base_dir, base_dir + '_Dataset.zip')

    URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_"

    download_and_extract(URL + 'text.zip', text_zip_path, base_dir)

    download_and_extract(URL + 'Dataset.zip', dataset_zip_path, base_dir)
