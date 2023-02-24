import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer

from ..builder import BACKBONES
import ipdb
import clip


@BACKBONES.register_module()
class clip_image(nn.Module):
    def __init__(self):
        super(clip_image, self).__init__()
        self.clip_model, self.preprocess = clip.load('RN50')
        self.clip_model.cuda().eval().float().requires_grad_(False)
        
        # for param in self.clip_model.parameters():
        #     param.requires_grad = False
        
    def forward(self, x):
        ipdb.set_trace()
        def stem(x):
            x = self.clip_model.visual.relu1(self.clip_model.visual.bn1(self.clip_model.visual.conv1(x)))
            x = self.clip_model.visual.relu2(self.clip_model.visual.bn2(self.clip_model.visual.conv2(x)))
            x = self.clip_model.visual.relu3(self.clip_model.visual.bn3(self.clip_model.visual.conv3(x)))
            x = self.clip_model.visual.avgpool(x)
            return x
        outs = []
        x = x.type(self.clip_model.visual.conv1.weight.dtype)
        x = stem(x)
        x = self.clip_model.visual.layer1(x)
        outs.append(x)
        x = self.clip_model.visual.layer2(x)
        outs.append(x)
        x = self.clip_model.visual.layer3(x)
        outs.append(x)
        x = self.clip_model.visual.layer4(x)
        outs.append(x)
      
        return tuple(outs)