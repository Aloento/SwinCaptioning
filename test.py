from FlickrDataset import FlickrDataset
from prepare import prepare

if __name__ == '__main__':
    des, vocab = prepare()
    print(len(des))
    print(len(vocab))

    dataset = FlickrDataset(des, vocab, "dev")
    print(len(dataset))
