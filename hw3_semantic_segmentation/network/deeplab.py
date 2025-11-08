import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3"]


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__()
        # TODO Problem 2.1
        # ================================================================================ #
        # ASPPConv: 3x3 convolution with atrous/dilated convolution
        # followed by batch normalization and ReLU activation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, 
                     dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # ================================================================================ #
    
    def forward(self, x):
        return self.conv(x)

class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        # TODO Problem 2.1
        # ================================================================================ #
        # ASPPPooling: Image-level features using global average pooling
        # Followed by 1x1 convolution, batch normalization, and ReLU
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling to 1x1
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # ================================================================================ #

    def forward(self, x):
        # TODO Problem 2.1
        # ================================================================================ #
        size = x.shape[-2:]  # (H, W)
        x = self.pool(x)
        # Upsample back to original spatial size using bilinear interpolation
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        # ================================================================================ #


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        # TODO Problem 2.1
        # ================================================================================ #
        # ASPP combines multiple parallel branches with different receptive fields:
        # 1. One 1x1 convolution
        # 2. Three 3x3 atrous convolutions with different dilation rates
        # 3. One image pooling branch
        
        out_channels = 256  # output channels for ASPP
        modules = []
        
        # 1x1 convolution branch
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # 3 atrous convolution branches with different dilation rates
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        # image pooling branch
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        # project concatenated features to final output channels
        # total input channels = 5 branches × 256 = 1280
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        # ================================================================================ #

    def forward(self, x):
        # TODO Problem 2.1
        # ================================================================================ #
        # apply all parallel branches and concatenate their outputs
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)  # concatenate along channel dimension
        
        # Project to final output
        return self.project(res)
        # ================================================================================ #


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()
        # TODO Problem 2.2
        # The model should have the following 3 arguments
        #   in_channels: number of input channels
        #   num_classes: number of classes for prediction
        #   aspp_dilate: atrous_rates for ASPP
        #   
        # ================================================================================ #
        # DeepLabV3 Head: Simple decoder with ASPP + classifier
        self.aspp = ASPP(in_channels, aspp_dilate)
        
        #  3x3 conv followed by 1x1 conv to num_classes
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        # ================================================================================ #
        
        self._init_weight()

    def forward(self, feature):
        # TODO Problem 2.2
        # ================================================================================ #
        # feature is an OrderedDict with 'out' key containing high-level features
        x = feature['out']
        x = self.aspp(x)
        x = self.classifier(x)
        return x
        # ================================================================================ #

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        # TODO Problem 2.2
        # The model should have the following 4 arguments
        #   in_channels: number of input channels
        #   low_level_channels: number of channels for project
        #   num_classes: number of classes for prediction
        #   aspp_dilate: atrous_rates for ASPP
        #   
        # ================================================================================ #
        # DeepLabV3+ Head: Encoder-decoder with ASPP + low-level feature fusion
        
        # ASPP module on high-level features (encoder output)
        self.aspp = ASPP(in_channels, aspp_dilate)
        
        # Project low-level features to reduce channels (from 256 to 48)
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder: refine features after concatenation
        # Input: 256 (from ASPP) + 48 (from low-level) = 304 channels
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        # ================================================================================ #
        self._init_weight()

    def forward(self, feature):
        # TODO Problem 2.2
        # ================================================================================ #
        # feature is an OrderedDict with 'out' (high-level) and 'low_level' keys
        
        # Process low-level features (from layer1, 1/4 resolution)
        low_level_feat = self.project(feature['low_level'])
        
        # Process high-level features with ASPP (from layer4, 1/16 resolution)
        output_feature = self.aspp(feature['out'])
        
        # Upsample high-level features 4x to match low-level feature resolution
        output_feature = F.interpolate(output_feature, size=low_level_feat.shape[2:], 
                                      mode='bilinear', align_corners=False)
        
        # Concatenate upsampled high-level features with projected low-level features
        concat_features = torch.cat([output_feature, low_level_feat], dim=1)
        
        # Pass through decoder to get final predictions
        return self.classifier(concat_features)
        # ================================================================================ #
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
