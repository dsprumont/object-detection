# common dependencies
import logging
import numpy as np
import torch
from torch import nn

# Detectron2 dependencies
from detectron2.structures import ImageList
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import build_backbone



@META_ARCH_REGISTRY.register()
class FeatureExtractor(nn.Module):
    """
    This meta architecture represents any kind of feature extractor.
    the purpose is to be able to train backbone with detectron2.

    It uses CrossEntropyLoss with mean reduction as criterion.
    """
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.num_classes = cfg.MODEL.BACKBONE.NUM_CLASSES
        self.classifier_name = cfg.MODEL.BACKBONE.CLASSIFIER
        self.criterion = nn.CrossEntropyLoss(reduction="mean") # good for now
        self.input_format = cfg.INPUT.FORMAT

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
    

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * label (optional): int, groundtruth class
                
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.

        Returns:
            list[dict]:
                Each dict is the output for one input image with the following key: "pred_classes"
        """

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_batchedimages(batched_inputs)
        assert "label" in batched_inputs[0], "No label found in batch"
        gt_instances = self.make_gt_tensor(batched_inputs)

        output = self.backbone(images.tensor)

        losses = {"loss_cls": self.criterion(output[self.classifier_name], gt_instances)}
        return losses

    
    def make_gt_tensor(self, batched_inputs):
        dims = (len(batched_inputs),)
        gt_instances = torch.zeros(dims, dtype=torch.long)
        for k, x in enumerate(batched_inputs):
            gt_instances[k] = x["label"]
        return gt_instances.to(self.device)



    def preprocess_batchedimages(self, batched_inputs):
        """
        Preprocess batch: normalized, resize, pad -> uniform batch
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    
    def inference(self, batched_inputs):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_batchedimages(batched_inputs)
        pred_logits = self.backbone(images.tensor)

        return pred_logits[self.classifier_name]
