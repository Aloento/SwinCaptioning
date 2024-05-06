让我们来实现一个基于 Swin Trasnformer 的 ImageCaptioning model。我们将使用 Flickr8k 数据集 和 Pytorch 及其相关库来训练我们的模型。

我现在已经有了 idx_to_word, train_loader, val_loader, test_loader
请帮我实现 对 torchvision.models.swin_v2_s 的调用 （Encoder class）
随后帮我实现对 MultiheadAttention 的调用 （Attention class）
然后帮我实现对 TransformerDecoderLayer 的调用 （Decoder class）
最后定义一个 Model 类，将上述三个类组合在一起，实现一个完整的 ImageCaptioning model。
