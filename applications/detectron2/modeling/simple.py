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
    Add config for simple resnet.
    """
    _C = cfg
    _C.MODEL.BACKBONE.NUM_CLASSES = 2
    _C.MODEL.BACKBONE.CLASSIFIER = "linear"


# highly inspired from
# https://github.com/facebookresearch/detectron2/blob/2292cb3f24cea3d5f77f36dbd8aaf1b8d8d14e08/detectron2/modeling/backbone/resnet.py
class BasicStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, norm="BN"):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__()
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3, # change from 7 to 3
            stride=1, # change from 2 to 1: makes output_stride lower 4 -> 2
            padding=1, # change from 3 to 1: keeps ouput_size = input_size
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

    @property
    def out_channels(self):
        return self.conv1.out_channels

    @property
    def stride(self):
        return 2  # = stride 1 conv -> stride 2 max pool


# highly inspired from
# https://github.com/facebookresearch/detectron2/blob/2292cb3f24cea3d5f77f36dbd8aaf1b8d8d14e08/detectron2/modeling/backbone/resnet.py
@BACKBONE_REGISTRY.register()
def build_simple_resnet_backbone(cfg, input_shape: ShapeSpec):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    # fmt: off
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    num_classes         = cfg.MODEL.BACKBONE.NUM_CLASSES if "linear" in out_features else None
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        14: [1, 1, 1, 1], # added configuration: simple Resnet (1 block per level + in/FC)
    }[depth]

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5, "linear": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
            "bottleneck_channels": bottleneck_channels,
            "stride_in_1x1": stride_in_1x1,
            "dilation": dilation,
            "num_groups": num_groups,
        }
        if deform_on_per_stage[idx]:
            stage_kargs["block_class"] = DeformBottleneckBlock
            stage_kargs["deform_modulated"] = deform_modulated
            stage_kargs["deform_num_groups"] = deform_num_groups
        else:
            stage_kargs["block_class"] = BottleneckBlock
        blocks = make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)
    return ResNet(stem, stages, num_classes=num_classes, out_features=out_features)


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
    bottom_up = build_simple_resnet_backbone(cfg, input_shape)
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