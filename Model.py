import torch
from torch import nn
from torchvision import models


class Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(Model, self).__init__()

        embed_dim = 256
        num_heads = 8
        num_decoder_layers = 6
        forward_expansion = 4

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

        self.swin = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)
        self.swin.head = nn.Linear(self.swin.head.in_features, embed_dim)
        self.relu = nn.LeakyReLU()

        self.attention = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * forward_expansion,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(self.attention, num_layers=num_decoder_layers)

        self.output_layer = nn.Linear(in_features=embed_dim, out_features=vocab_size)

    def forward(self, images, captions):
        images = self.swin(images)  # (batch_size, embed_dim)
        images = self.relu(images)
        images = images.unsqueeze(1)  # (batch_size, seq_len 1, embed_dim)

        captions = self.embedding(captions)  # (batch_size, seq_len 20, embed_dim)

        cap_mask = torch.triu(torch.ones(captions.size(1), captions.size(1)), diagonal=1).bool()
        cap_mask = cap_mask.to(captions.device)

        output = self.decoder(
            captions,
            images,
            tgt_mask=cap_mask
        )

        output = self.output_layer(output)
        return output


if __name__ == '__main__':
    model = Model(1000)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    x = torch.rand((2, 3, 256, 256))
    y = torch.randint(0, 1000, (2, 20))

    out = model(x, y)
    print(out.shape)
