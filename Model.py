import torch
from torch import nn
from torchvision import models


class ImageCaptioningModel(nn.Module):
    def __init__(self, num_heads, num_decoder_layers, forward_expansion, vocab_size: int):
        super(ImageCaptioningModel, self).__init__()
        embed_dim = 256

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

        self.swin = models.swin_v2_s(pretrained=True)
        self.swin.head = nn.Linear(self.swin.head.in_features, embed_dim)
        self.relu = nn.LeakyReLU()

        self.attention = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * forward_expansion
        )

        self.decoder = nn.TransformerDecoder(self.attention, num_layers=num_decoder_layers)

        self.output_layer = nn.Linear(in_features=embed_dim, out_features=vocab_size)

    def forward(self, images, captions):
        images = self.swin(images)
        images = self.relu(images)

        batch_size, feature_dim, h, w = images.size()
        images = images.view(batch_size, feature_dim, h * w)
        images = images.permute(2, 0, 1)

        captions = self.embedding(captions)

        cap_mask = torch.triu(torch.ones(captions.size(0), captions.size(0)), diagonal=1).bool()
        cap_mask = cap_mask.to(captions.device)

        output = self.decoder(
            captions,
            images,
            tgt_mask=cap_mask
        )

        output = self.output_layer(output)
        return output


if __name__ == '__main__':
    model = ImageCaptioningModel(8, 6, 4, 10000)
    print(model)
