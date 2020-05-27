import math
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import torch
from torch import nn

from detectron2.layers import (
    Conv2d,
     DeformConv,
    FrozenBatchNorm2d,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm
)
from detectron2.modeling import (
    BACKBONE_REGISTRY, 
    Backbone, 
    ShapeSpec,
    build_backbone,
    FPN
)
from detectron2.modeling.backbone.resnet import (
    ResNet,
    ResNetBlockBase,
    BottleneckBlock,
    DeformBottleneckBlock,
    make_stage
)


# ----------- EXPERIMENTAL -------------- #
def chstr(string, str2rm=None):
    if str2rm is None:
        return string
    substrings = string.split(str2rm)
    return '.'.join(substrings)

def addstr(string, str2add=None):
    if str2add is None:
        return string
    substrings = string.split("backbone.")
    return ("backbone"+str2add+substrings[1])


def load_backbone_weights(model, path):
    """
    Load the Simple_resnet weights into the mark_rcnn_fpn detector.

    Because it uses a FPN as feature extractor, the Resnet model is contained
    as a subpart of the FPN in the .bottom_up. module. However as we trained the simple_resnet
    alone, it is "contained at the root ." This bit of code looks for correspondence by 
    adding the ".bottom_up" keyword to the name of each simple_resnet modules.
    """
    # 0. get state_dicts
    model_state_dict = model.state_dict()
    print('------------------------------------- MODEL ---------------------------------')
    for k in model_state_dict.keys():
        print(k)
    loaded_dict = torch.load(path)['model']
    print('------------------------------------ WEIGHTS --------------------------------')
    for k in loaded_dict.keys():
        print(k)
    # 1. filter out unnecessary keys
    pretrained_dict = {addstr(k, '.bottom_up.'): v for k, v in loaded_dict.items() if addstr(k, '.bottom_up.') in model_state_dict}
    print('--------------------------------- PRETRAINED --------------------------------')
    for k in pretrained_dict.keys():
        print(k)
    # 2. overwrite entries in the existing state dict
    model_state_dict.update(pretrained_dict) 
    print('-------------------------------- UPDATED MODEL ------------------------------')
    for k in model_state_dict.keys():
        print(k)
    # 3. load the new state dict
    model.load_state_dict(model_state_dict)

# ----------- EXPERIMENTAL -------------- #



def add_simple_resnet_config(cfg):
    """
    Add config for resnet-like model.
    """
    _C = cfg
    _C.MODEL.BACKBONE.NUM_CLASSES = 2
    _C.MODEL.BACKBONE.CLASSIFIER = "logits"
    _C.MODEL.RESNETS.BLOCKS_PER_STAGE = [1, 1, 1, 1]
    _C.MODEL.RESNETS.STEM_KERNEL = 7
    _C.MODEL.RESNETS.STEM_STRIDE = 2
    _C.MODEL_RESNETS.STEM_POOLING_ON = True


# highly inspired from
# https://github.com/facebookresearch/detectron2/blob/2292cb3f24cea3d5f77f36dbd8aaf1b8d8d14e08/detectron2/modeling/backbone/resnet.py
class StemBlock(nn.Module):
    """
    First block in a Resnet architecture.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=7,
        stride=2,
        pooling=nn.MaxPool2d,
        norm="BN"
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # args stride is for the conv layer,
        # self.stride is for the whole module (stride + maxpool)
        self.stride = stride*2
        self.kernel_size = kernel_size
        self.pooling = pooling

        self.conv1 = Conv2dWithNorm(
            in_channels,
            out_channels,
            kernel_size=kernel_size,  # Original is 7 /!\
            stride=stride,
            padding=int((kernel_size-1)/2),  # Original is 3 /!\
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        if pooling is not None:
            self.maxpool = pooling(kernel_size=3, stride=2, padding=1)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')

    def forward(self, x):
        h = self.conv1(x)
        h = self.relu(h)
        if self.pooling is not None:
            h = self.maxpool(h)
        return h

    @property
    def out_channels(self):
        return self.conv1.out_channels

    @property
    def stride(self):
        return 2  # = stride 1 conv -> stride 2 max pool


class ResNetLike(nn.Module):

    def __init__(self, stem, blocks, num_classes, out_features=['logits'], freeze_at=0):
        super().__init__()
        self.stem = stem
        self.out_features = out_features
        self.out_feature_channels = {'stem': stem.out_channels}
        self.out_feature_stride = {'stem': stem.stride}

        last_channels = 0
        curr_stride = stem.stride
        self.res = []
        for n, block in enumerate(blocks):
            layers = []
            name = 'res'+str(n+2)
            in_channels = block['in_channels']
            for idx in range(block['count']):
                stride = 1 if (idx > 0 or block['dilation'] > 1) else 2
                out_channels = block['out_channels']
                layers.append(
                    Bottleneck(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        bottleneck_channels=block['bottleneck_channels'],
                        stride=stride,
                        dilation=block['dilation']
                    )
                )
                in_channels = out_channels
                last_channels = out_channels
            curr_stride = curr_stride * 2
            self.out_feature_channels[name] = last_channels
            self.out_feature_stride[name] = curr_stride

            if freeze_at >= stage_idx:
                for layer in layers:
                    layer.freeze()

            module = nn.Sequential(*layers)
            self.res.append([name, module])
            self.add_module(name, module)

        if 'logits' in out_features:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(last_channels, num_classes)
            nn.init.normal_(self.linear.weight, std=0.01)

    def forward(self, x):
        out = {}
        h = self.stem(x)
        if 'stem' in self.out_features:
            out['stem'] = h
        for name, res in self.res:
            h = res(h)
            if name in self.out_features:
                out[name] = h
        if 'logits' in self.out_features:
            h = self.avgpool(h)
            h = torch.flatten(h, 1)  # flatten all dimensions, except batch dim
            h = self.linear(h)
            out['logits'] = h
        return out

    def stage_outputs(self):
        # print(self.out_features)
        return {name: {
            'stride': self.out_feature_stride[name],
            'channels': self.out_feature_channels[name]}
            for name in self.out_features
            }

    @staticmethod
    def build(name, input_channels, num_classes, out_features=['logits']):
        cfg = CONFIGURATIONS.get(name, None)
        if cfg is None:
            raise AttributeError(
                "The given resnet configuration is not available.")
            sys.exit()

        in_channels = input_channels
        bottleneck_channels: int = cfg['stem_out_channels']
        out_channels: int = cfg['res2_out_channels']

        stem = StemBlock(
            in_channels=in_channels,
            out_channels=cfg['stem_out_channels'],
            stride=cfg['stem_stride'],
            kernel_size=cfg['stem_kernel'],
            pooling=cfg['stem_pooling']
        )

        blocks = []
        in_channels = cfg['stem_out_channels']
        bottleneck_channels = cfg['res2_bottleneck_channels']
        out_channels = cfg['res2_out_channels']
        blocks_per_layer = cfg['resnet_layers']
        dilations = cfg['resnet_dilations']

        for idx in range(len(blocks_per_layer)):
            block = {
                'count': blocks_per_layer[idx],
                'in_channels': in_channels,
                'bottleneck_channels': bottleneck_channels,
                'out_channels': out_channels,
                'dilation': dilations[idx]
            }
            in_channels = out_channels
            bottleneck_channels *= 2
            out_channels *= 2
            blocks.append(block)

        return ResNetLike(stem, blocks, num_classes, out_features)


# highly inspired from
# https://github.com/facebookresearch/detectron2/blob/2292cb3f24cea3d5f77f36dbd8aaf1b8d8d14e08/detectron2/modeling/backbone/resnet.py
@BACKBONE_REGISTRY.register()
def build_resnetlike_backbone(cfg, input_shape: ShapeSpec):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """
        # fmt: off
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    num_classes         = cfg.MODEL.BACKBONE.NUM_CLASSES if "logits" in out_features else None
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    num_blocks_per_stage = cfg.MODEL.RESNETS.BLOCKS_PER_STAGE
    dilation_per_stage = cfg.MODEL.RESNETS.DILATION_PAR_STAGE
    norm = cfg.MODEL.RESNETS.NORM
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    # fmt: on
    
    # need registration of new blocks/stems?
    stem = StemBlock(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        kernel_size=cfg.MODEL.RESNETS.STEM_KERNEL,
        stride=cfg.MODEL.RESNETS.STEM_STRIDE,
        pooling=nn.MaxPool2d if cfg.MODEL.RESNETS.STEM_POOLING_ON == True else None,
        norm=norm,
    )  

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5, "logits": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)

    blocks = []
    for idx in range(2, max_stage_idx + 1):
        block = {
            'count': num_blocks_per_stage[idx],
            'in_channels': in_channels,
            'bottleneck_channels': bottleneck_channels,
            'out_channels': out_channels,
            'dilation': dilation_per_stage[idx]
        }
        in_channels = out_channels
        bottleneck_channels *= 2
        out_channels *= 2
        blocks.append(block)

    return ResNetLike(stem, stages, num_classes=num_classes, out_features=out_features, freeze_at=0)


# taken from
# https://github.com/facebookresearch/detectron2/blob/2292cb3f24cea3d5f77f36dbd8aaf1b8d8d14e08/detectron2/modeling/backbone/fpn.py
class LastLevelMaxPool(nn.Module):
    """
    P5 feature from P4. (Originaly, P6 from P5 but our stem has stride 2 instead of 4)
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p4"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


# highly inspired from
# https://github.com/facebookresearch/detectron2/blob/2292cb3f24cea3d5f77f36dbd8aaf1b8d8d14e08/detectron2/modeling/backbone/fpn.py
@BACKBONE_REGISTRY.register()
def build_simple_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnetlike_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone