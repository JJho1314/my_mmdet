import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer

from ..builder import BACKBONES
import ipdb
import clip
from mmcv.runner import BaseModule

from torch.nn import functional as F

from collections import OrderedDict
from typing import Tuple, Union

import logging
import torch
import torch.distributed as dist

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()

@BACKBONES.register_module()
class clip_image(nn.Module):
    def __init__(self):
        super(clip_image, self).__init__()
        self.clip_model, self.preprocess = clip.load('RN50')
        # self.clip_model.cuda().eval().float().requires_grad_(False)
        # self.clip_model.apply(fix_bn)

        # for param in self.clip_model.parameters():
        #     param.requires_grad = False
        
    def forward(self, x):
        # ipdb.set_trace()
        def stem(x):
            x = self.clip_model.visual.relu1(self.clip_model.visual.bn1(self.clip_model.visual.conv1(x)))
            x = self.clip_model.visual.relu2(self.clip_model.visual.bn2(self.clip_model.visual.conv2(x)))
            x = self.clip_model.visual.relu3(self.clip_model.visual.bn3(self.clip_model.visual.conv3(x)))
            x = self.clip_model.visual.avgpool(x)
            return x
        # self.clip_model.apply(fix_bn)

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

class FrozenBatchNorm2d(BaseModule):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.

    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.

    Other pre-trained backbone models may contain all 4 parameters.

    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3

    def __init__(self, num_features, eps=1e-5, init_cfg=None):
        super(FrozenBatchNorm2d, self).__init__(init_cfg)
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype  # may be half
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # when use offline modules, avoid overwriting running mean and var for loaded weights
            skip_reset = False
            for k_n in state_dict: # checkpoint weights
                if 'ignore_others' in k_n: #if 'offline' in k_n:
                    skip_reset = True
            if not skip_reset:
                # No running_mean/var in early versions
                # This will silent the warnings
                if prefix + "running_mean" not in state_dict:
                    state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
                if prefix + "running_var" not in state_dict:
                    state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        # NOTE: if a checkpoint is trained with BatchNorm and loaded (together with
        # version number) to FrozenBatchNorm, running_var will be wrong. One solution
        # is to remove the version number from the checkpoint.
        if version is not None and version < 3:
            logger = logging.getLogger(__name__)
            logger.info("FrozenBatchNorm {} is upgraded to version 3.".format(prefix.rstrip(".")))
            # In version < 3, running_var are used without +eps.
            state_dict[prefix + "running_var"] -= self.eps

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res
    
class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, init_cfg = None):
        super(Bottleneck, self).__init__(init_cfg)

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = FrozenBatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = FrozenBatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = FrozenBatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", FrozenBatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out

    
@BACKBONES.register_module()
class ModifiedResNet(BaseModule):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, freeze_at, input_resolution=224, width=64, init_cfg=None):
        super(ModifiedResNet,self).__init__(init_cfg)
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = FrozenBatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = FrozenBatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = FrozenBatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        
        self.freeze(freeze_at)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        out = []
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        out.append(x)
        x = self.layer2(x)
        out.append(x)
        x = self.layer3(x)
        out.append(x)
        x = self.layer4(x)
        out.append(x)

        return tuple(out)
    
    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ResNet itself
        """
        def cnnblockbase_freeze(nn_module):
            """
            Make this block not trainable.
            This method sets all parameters to `requires_grad=False`,
            and convert all BatchNorm layers to FrozenBatchNorm

            Returns:
                the block itself
            """
            for p in nn_module.parameters():
                p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(nn_module)
        
        if freeze_at >= 1: # stem
            cnnblockbase_freeze(self.conv1)
            cnnblockbase_freeze(self.bn1)
            cnnblockbase_freeze(self.conv2)
            cnnblockbase_freeze(self.bn2)
            cnnblockbase_freeze(self.conv3)
            cnnblockbase_freeze(self.bn3)
        # each stage is a torch.nn.modules.container.Sequential
        for idx, stage in enumerate([self.layer1, self.layer2, self.layer3, self.layer4], start=2): 
            if freeze_at >= idx:
                for block in stage.children():  # each block is a Bottleneck
                    cnnblockbase_freeze(block)  
        return self
 
@BACKBONES.register_module()    
def ModifiedResNet50(layers, freeze_at):
    
    resnet50 = ModifiedResNet(layers, freeze_at)
    
    net_dict = resnet50.state_dict()
    # ipdb.set_trace()
    pretrained_model = torch.jit.load('/home/work/.cache/clip/RN50.pt',map_location=torch.device('cpu'))
    
    state_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in net_dict.keys()}
    net_dict.update(state_dict)
    resnet50.load_state_dict(net_dict)
    
    return resnet50

