# common dependencies
import datetime
import logging
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from contextlib import contextmanager

# Detectron2 dependencies
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds, create_small_table
from detectron2.evaluation import DatasetEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog


class BinaryClassificationEvaluator(DatasetEvaluator):
    """
    An evaluator designed for binary classification task.

    In the provided datasets, labels 0 corresponds to negative class.
    """
    
    def __init__(self, dataset_name, output_dir=None, validate=False):

        self.output_dir = output_dir
        self.validate = validate

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger("detectron2") # instead of __name__

        self._metadata = MetadataCatalog.get(dataset_name)
        assert len(self._metadata.thing_classes) == 2, \
            "BinaryClassificationEvaluator only deals with 2-classes datasets"


    def reset(self):
        self._predictions = {
            'gt_cls': [],
            'pred_cls': []
        }
        if self.validate:
            self._predictions['logits'] = []


    def process(self, inputs, outputs):
        """
        Accumulate ground_truth and predicted labels.
        """
        for input, output in zip(inputs, outputs):
            self._predictions['gt_cls'].append(input['label'])
            if self.validate:
                self._predictions['logits'].append(output.to(self._cpu_device))
            pred_logits = output.to(self._cpu_device).numpy()
            pred_cls = np.argmax(pred_logits)
            self._predictions['pred_cls'].append(pred_cls)


    def evaluate(self):
        """
        Compute evaluation metrics based on accumulated data.

        Returns:
            dict: keys are [ErrorRate, Accuracy, Precision, Recall, Specificity]
        """

        predictions = self._predictions

        if len(predictions['gt_cls']) == 0:
            self._logger.warning("[BinaryClassificationEvaluator] Did not receive valid predictions.")
            return {}

        if self.validate:
            pred_tensor = torch.stack(predictions['logits'], dim=0)
            gt_tensor = torch.LongTensor(predictions['gt_cls'])
            loss = nn.CrossEntropyLoss()(pred_tensor, gt_tensor)

        # if self._output_dir:
        #     PathManager.mkdirs(self._output_dir)
        #     file_path = os.path.join(self._output_dir, "class_predictions.pth")
        #     with PathManager.open(file_path, "wb") as f:
        #         torch.save(predictions, f)

        gt_ = np.array(predictions['gt_cls'])
        pred_ = np.array(predictions['pred_cls'])
        
        total = len(gt_)

        tp_ = np.logical_and(                gt_,                 pred_) # gt_ = 1 AND pred_ = 1
        fp_ = np.logical_and(np.logical_not(gt_),                 pred_) # gt_ = 0 AND pred_ = 1
        fn_ = np.logical_and(gt_                , np.logical_not(pred_)) # gt_ = 1 AND pred_ = 0
        tn_ = np.logical_and(np.logical_not(gt_), np.logical_not(pred_)) # gt_ = 0 AND pred_ = 0
        
        tp = np.sum(tp_)
        fp = np.sum(fp_)
        fn = np.sum(fn_)
        tn = np.sum(tn_)

        P = tp + fn
        N = tn + fp

        self._results = OrderedDict()
        class_names = self._metadata.thing_classes # Indicate which class is Positive
        
        self._results['ErrorRate'] = (fp + fn) / ((N + P) + 1e-5)
        self._results['Accuracy'] = (tp + tn) / ((N + P) + 1e-5)
        self._results['Precision'] = tp / ((tp + fp) + 1e-5)
        self._results['Recall'] = tp / (P + 1e-5)
        self._results['Specificity'] = 1 - fp / (N + 1e-5)

        results = {
            key: float(value * 100 if self._results[key] >= 0 else "nan")
            for key, value in self._results.items()
        }
        results[class_names[1]+'(P)'] = P
        results[class_names[0]+'(N)'] = N

        self._results[class_names[1]+'(P)'] = P
        self._results[class_names[0]+'(N)'] = N

        if self.validate:
            self._results['loss_cls'] = loss

        self._logger.info(
            "Evaluation results for classification: \n" + create_small_table(results)
        )
        if not np.isfinite(sum(self._results.values())):
            self._logger.info("Note that some metrics cannot be computed.")

        return copy.deepcopy(self._results)

