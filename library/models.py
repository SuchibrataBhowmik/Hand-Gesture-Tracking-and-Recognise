
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, zeros_, normal_


class BaselineEmbeddingNet(nn.Module):
    def __init__(self):
        super(BaselineEmbeddingNet, self).__init__()
        self.fully_conv = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=2, bias=True),
                                        nn.BatchNorm2d(96),
                                        nn.ReLU(),
                                        nn.MaxPool2d(3, stride=2),

                                        nn.Conv2d(96, 256, kernel_size=5, stride=1, groups=2, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.MaxPool2d(3, stride=1),

                                        nn.Conv2d(256, 384, kernel_size=3, stride=1, groups=1, bias=True),
                                        nn.BatchNorm2d(384),
                                        nn.ReLU(),

                                        nn.Conv2d(384, 384, kernel_size=3, stride=1, groups=2, bias=True),
                                        nn.BatchNorm2d(384),
                                        nn.ReLU(),
                                        
                                        nn.Conv2d(384, 32, kernel_size=3, stride=1, groups=2, bias=True))

    def forward(self, x):
        output = self.fully_conv(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class VGG11EmbeddingNet_5c(nn.Module):
    def __init__(self):
        super(VGG11EmbeddingNet_5c, self).__init__()
        self.fully_conv = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=True),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, stride=2),

                                        nn.Conv2d(64, 128, kernel_size=3, stride=1, bias=True),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, stride=2),
                                        nn.Conv2d(128, 256, kernel_size=3, stride=1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 256, kernel_size=3, stride=1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        # Added ConvLayer, not in original model
                                        nn.Conv2d(256, 32, kernel_size=3, stride=1, bias=True))

    def forward(self, x):
        output = self.fully_conv(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class VGG16EmbeddingNet_8c(nn.Module):
    def __init__(self):
        super(VGG16EmbeddingNet_8c, self).__init__()
        self.fully_conv = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=True),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=True),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, stride=2),
                                        nn.Conv2d(64, 128, kernel_size=3, stride=1, bias=True),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=True),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, stride=2),
                                        nn.Conv2d(128, 256, kernel_size=3, stride=1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 256, kernel_size=3, stride=1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 256, kernel_size=3, stride=1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        # Added ConvLayer, not in original model
                                        nn.Conv2d(256, 32, kernel_size=3, stride=1, bias=True))

    def forward(self, x):
        output = self.fully_conv(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class SiameseNet(nn.Module):
    def __init__(self, embedding_net, upscale=False, corr_map_size=33, stride=4):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.match_batchnorm = nn.BatchNorm2d(1)
        self.upscale = upscale
        self.corr_map_size = corr_map_size
        self.stride = stride
        self.upsc_size = (self.corr_map_size-1)*self.stride + 1
        
        if upscale:
            self.upscale_factor = 1
        else:
            self.upscale_factor = self.stride

    def forward(self, x1, x2):
        embedding_reference = self.embedding_net(x1)
        embedding_search = self.embedding_net(x2)
        match_map = self.match_corr(embedding_reference, embedding_search)
        return match_map

    def get_embedding(self, x):
        return self.embedding_net(x)

    def match_corr(self, embed_ref, embed_srch):
        b, c, h, w = embed_srch.shape
        match_map = F.conv2d(embed_srch.view(1, b * c, h, w), embed_ref, groups=b)
        
        match_map = match_map.permute(1, 0, 2, 3)
        match_map = self.match_batchnorm(match_map)
        if self.upscale:
            match_map = F.interpolate(match_map, self.upsc_size, mode='bilinear', align_corners=False)
        
        return match_map


def weights_init(model):
    if isinstance(model, nn.Conv2d):
        xavier_uniform_(model.weight, gain=math.sqrt(2.0))
        constant_(model.bias, 0.1)
    elif isinstance(model, nn.BatchNorm2d):
        normal_(model.weight, 1.0, 0.02)
        zeros_(model.bias)
