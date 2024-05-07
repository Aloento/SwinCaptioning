import torch
from torcheval.metrics.functional import bleu_score
from tqdm import tqdm

from Model import Model
from persist import load_checkpoint
from prepare import prepare

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate():
    idx_to_word, _, _, test_loader = prepare(True)
    model = Model(len(idx_to_word))
    model = model.to(device)

    epoch = load_checkpoint(model)
    print(f"Model loaded from epoch {epoch}")

    model.eval()
    actual_sequences = []
    predicted_sequences = []

    with torch.no_grad():
        for images, captions, indices in tqdm(test_loader, desc='Evaluating Model', leave=False):
            images = images.to(device)
            indices = indices.to(device)

            outputs, _ = model(images, indices)

            # Decoding predictions to words
            predicted_indices = torch.argmax(outputs, dim=2)
            predicted_indices = predicted_indices.cpu().numpy()

            for true_caption, pred_idx in zip(captions, predicted_indices):
                pred_caption = [idx_to_word[index] for index in pred_idx]
                pred_caption = pred_caption[:pred_caption.index('[EOF]')] if '[EOF]' in pred_caption else pred_caption
                pred_caption = ' '.join(pred_caption)

                actual_sequences.append(true_caption.lower())
                predicted_sequences.append(pred_caption)

    bleu = bleu_score(predicted_sequences, actual_sequences)
    print(f"BLEU Score: {bleu:.2f}")
    return bleu


if __name__ == '__main__':
    evaluate()
