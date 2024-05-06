import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Model import Model
from prepare import prepare


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_epoch(
        model: Model,
        data_loader: DataLoader,
        optimizer: optim.AdamW,
        criterion: nn.CrossEntropyLoss,
        scaler: GradScaler,
        writer: SummaryWriter,
        epoch: int
):
    model.train()
    running_loss = 0.0
    total = 0

    loop = tqdm(data_loader, desc=f'Train Epoch {epoch}', leave=True)

    for batch_idx, (images, captions, indices) in enumerate(loop):
        images = images.to(device)
        indices = indices.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images, indices)

        loss = criterion(outputs.view(-1, outputs.size(2)), indices.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

        loop.set_postfix(loss=running_loss / total)
        current = epoch * len(data_loader) + batch_idx
        writer.add_scalar('Loss/train', loss.item(), current)

    return running_loss / total


def validate_epoch(
        model: Model,
        data_loader: DataLoader,
        criterion: nn.CrossEntropyLoss,
        writer: SummaryWriter,
        epoch: int
):
    model.eval()
    running_loss = 0.0
    total = 0

    loop = tqdm(data_loader, desc=f'Val Epoch {epoch}', leave=True)

    with torch.no_grad():
        for batch_idx, (images, captions, indices) in enumerate(loop):
            images = images.to(device)
            indices = indices.to(device)

            outputs = model(images, indices)

            loss = criterion(outputs.view(-1, outputs.size(2)), indices.view(-1))

            running_loss += loss.item() * images.size(0)
            total += images.size(0)

            loop.set_postfix(loss=running_loss / total)
            current = epoch * len(data_loader) + batch_idx
            writer.add_scalar('Loss/val', loss.item(), current)

    return running_loss / total


def run():
    idx_to_word, train_loader, val_loader, test_loader = prepare()

    model = Model(len(idx_to_word)).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    scaler = GradScaler()
    writer = SummaryWriter()

    loop = tqdm(range(10), desc='Epochs', leave=True)

    for epoch in loop:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, writer, epoch)
        val_loss = validate_epoch(model, val_loader, criterion, writer, epoch)

        scheduler.step(val_loss)

    writer.close()


if __name__ == '__main__':
    run()
