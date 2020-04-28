# common dependencies
import logging
import numpy as np
import torch
import itertools
import operator
from tabulate import tabulate
from termcolor import colored
import copy
from torch import nn

# Detectron2 dependencies
from detectron2.utils.comm import get_world_size
from detectron2.data import (
    DatasetFromList,
    MapDataset,
    DatasetMapper,
    DatasetCatalog,
    MetadataCatalog
)
from detectron2.utils.logger import log_first_n
from detectron2.data import samplers
from detectron2.data.common import AspectRatioGroupedDataset
from detectron2.data.build import trivial_batch_collator, worker_init_reset_seed
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T


# inspired from
# https://github.com/facebookresearch/detectron2/blob/a809b8670d29693076f28eeb4a94838072ccd0d9/detectron2/data/dataset_mapper.py
class ClsDatasetMapper():
    """
    A Dataset mapper that apply data augmentation in a classification task context.
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format = cfg.INPUT.FORMAT
        # fmt: on
        self.is_train = is_train


    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept (classification only)
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(
            ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
        )

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        return dataset_dict



# highly inspired from
# https://github.com/facebookresearch/detectron2/blob/a809b8670d29693076f28eeb4a94838072ccd0d9/detectron2/data/build.py
def build_classification_train_loader(cfg, mapper=None):
    """
    Build a classification data loader from cfg.

    Returns:
        list[dict]: Each dict contains,
        * image: Tensor, image in (C, H, W) format.
        * label (optional): int, groundtruth class
    """
    
    num_workers = get_world_size()
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    assert (
        images_per_batch % num_workers == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    assert (
        images_per_batch >= num_workers
    ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    images_per_worker = images_per_batch // num_workers

    dataset_dicts = get_classification_dataset_dicts(cfg.DATASETS.TRAIN)
    dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = ClsDatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        sampler = samplers.TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        sampler = samplers.RepeatFactorTrainingSampler(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    if cfg.DATALOADER.ASPECT_RATIO_GROUPING:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=None,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        data_loader = AspectRatioGroupedDataset(data_loader, images_per_worker)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_worker, drop_last=True
        )
        # drop_last so the batch always have the same size
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )

    return data_loader


def build_classification_test_loader(cfg, dataset_name, mapper=None):
    
    dataset_dicts = get_classification_dataset_dicts(cfg.DATASETS.TEST)
    dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = ClsDatasetMapper(cfg, False) # False means Not is_training
    dataset = MapDataset(dataset, mapper)

    sampler = samplers.InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def get_classification_dataset_dicts(dataset_names):

    # retrieve datasets from Catalog
    assert len(dataset_names)
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    # Merge all datasets into one
    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    try:
        class_names = MetadataCatalog.get(dataset_names[0]).thing_classes
        utils.check_metadata_consistency("thing_classes", dataset_names)
        print_instances_class_histogram(dataset_dicts, class_names)
    except AttributeError:  # class names are not available for this dataset
        pass
    
    return dataset_dicts


def print_instances_class_histogram(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for entry in dataset_dicts:
        label = [entry["label"]]
        histogram += np.histogram(label, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    log_first_n(
        logging.INFO,
        "Distribution of instances among all {} categories:\n".format(num_classes)
        + colored(table, "cyan"),
        key="message",
    )