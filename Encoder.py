from torch import nn
from torchvision import models


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        swin = models.swin_v2_s(weights=models.Swin_V2_S_Weights.DEFAULT)
        modules = list(swin.children())[:-1]
        self.swin = nn.Sequential(*modules)

        encoded_image_size = 14
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def forward(self, images):
        """
        (batch_size, 3, image_height, image_width) ->
        (batch_size, encoded_image_size, encoded_image_size, channels)
        """
        # 利用 Swin Transformer 提取特征
        out = self.swin(images)
        # 改变特征图的空间维度
        out = self.adaptive_pool(out)
        # 调整通道，以便特征图的通道数位于最后一个维度
        out = out.permute(0, 2, 3, 1)
        return out
