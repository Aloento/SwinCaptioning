from tqdm import tqdm

from prepare import prepare

if __name__ == '__main__':
    idx_to_word, train_loader, val_loader, test_loader = prepare()

    for i in tqdm(test_loader):
        print(i)
        break
