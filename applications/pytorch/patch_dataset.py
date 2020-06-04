import os
import math
import pydicom
import itertools
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from utils import compute_iou_matrix
from dataset import ToTensor


class CropTransform:
    """
    Take a PIL image and return a crop region as PIL image.
    """
    def __init__(self, top, left, width, height):
        self.width = height  # width
        self.height = width  # height
        self.top = left  # top
        self.left = top  # left

    def __call__(self, image):
        return TF.crop(image, self.top, self.left, self.height, self.width)


class PatchBasedDataset(Dataset):

    def __init__(
        self,
        path,
        subset,
        patch_size=256,
        mode='grayscale',
        bits=8,
        mean=[0.5, 0.5, 0.5],
        std=[1.0, 1.0, 1.0],
        training=False,
        filter_with_annotations=False
    ):
        self.path = path
        self.data_images = os.path.join(path, subset)
        self.data_labels = os.path.join(path, 'labels')

        if isinstance(bits, (int,)) and bits in (8, 16):
            self.bits_per_channel = bits
        else:
            raise ValueError("bits expected an integer value of 8 or 16,\
                 got {} instead".format(bits))

        if isinstance(mode, (str,)) and mode in ['grayscale', 'rgb']:
            if self.bits_per_channel == 16 and mode == 'rgb':
                raise ValueError("rgb mode is not supported for 16bits images")
            else:
                self.mode = {'grayscale': 'L', 'rgb': 'RGB'}[mode]
        else:
            raise ValueError("mode expected to be in [grayscale, rgb],\
                 got {} instead".format(mode))

        self.in_channels = 1 if mode == 'L' else 3
        self.training = training
        self.filter_with_annotations = filter_with_annotations
        self.patch_size = patch_size

        # either,
        # 8bits grayscale
        # 8bits rgb
        # 16bits grayscale
        if isinstance(mean, (int, float)):
            if self.mode == 'RGB':
                self.mean = (mean, mean, mean)
            else:
                self.mean = (mean,)
        elif isinstance(mean, (list,)) \
                and all(isinstance(m, (int, float,)) for m in mean):
            if self.mode == 'RGB':
                self.mean = mean
            else:
                raise ValueError("mean expected to be single value"
                                 " with grayscale mode, got tuple instead")
        else:
            raise ValueError("mean expected to be int, float or "
                             "list[int, float], got {} instead".format(mean))

        if isinstance(std, (int, float,)):
            if self.mode == 'RGB':
                self.std = (std, std, std)
            else:
                self.std = (std,)
        elif isinstance(std, (list,)) \
                and all(isinstance(m, (int, float)) for m in std):
            if self.mode == 'RGB':
                self.std = std
            else:
                raise ValueError("std expected to be single value"
                                 " with grayscale mode, got tuple instead")
        else:
            raise ValueError("std expected to be int, float or "
                             "list[int, float], got {} instead".format(std))

        images = os.listdir(self.data_images)
        images = [image for image in images
                  if image.rsplit('.')[1].lower()
                  in ['jpg', 'jpeg', 'png', 'dcm']]
        self.image_files = images

        labels = []
        for image in self.image_files:
            label = image.replace('.jpg', '.txt').replace(
                '.jpeg', '.txt').replace('.png', '.txt').replace(
                    '.dcm', '.txt')
            labels.append(label)
        self.labels = labels

        gt_boxes = []
        for k, filename in enumerate(labels):
            with open(os.path.join(self.data_labels, filename)) as f:
                for line in f.readlines():
                    words = line.split(' ')
                    assert(len(words) == 5)
                    # category = words[0]
                    xmin = int(words[1])
                    xmax = int(words[2])
                    ymin = int(words[3])
                    ymax = int(words[4])
                    gt_boxes.append(
                        [xmin, ymin, xmax, ymax,
                         (xmax-xmin)*(ymax-ymin), k])
        self.gt_boxes = gt_boxes

        self.image_sizes = _get_image_dimensions(
            self.data_images, self.image_files,
            self.mode, self.bits_per_channel)
        self.patches, self.patch_grids = _get_patches_coords(
            self.image_sizes, patch_size)

        if self.filter_with_annotations:
            self.positive_patches, self.gt_box_positive, self.gt_boxes\
                 = _filter_positive_samples(
                    self.patches, self.gt_boxes,
                    len(self.image_files), iou_threshold=0.7)
            self.patches = self.positive_patches
        else:
            self.positive_patches, self.gt_box_positive, self.gt_boxes\
                = _associate_bbox_to_patch(
                    self.patches, self.gt_boxes,
                    len(self.image_files), iou_threshold=0.7)
            self.patches = self.positive_patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        # print(len(self.gt_box_positive))
        patch = self.patches[index]
        image = _load_image(self.data_images, self.image_files[patch[5]],
                            bits=self.bits_per_channel, mode=self.mode)

        trsf_crop = CropTransform(
            patch[0], patch[1], self.patch_size, self.patch_size)
        trsf_hflip = transforms.RandomHorizontalFlip(p=0.5)
        trsf_tensor = transforms.ToTensor() \
            if self.bits_per_channel == 8 else ToTensor()
        trsf_normalize = transforms.Normalize(self.mean, self.std)

        trsf_compose = [trsf_crop]
        if self.training is True:
            trsf_compose.append(trsf_hflip)
        trsf_compose.append(trsf_tensor)
        trsf_compose.append(trsf_normalize)

        data_transforms = transforms.Compose(trsf_compose)
        image = data_transforms(image)

        bboxes = self.gt_box_positive[index]

        pad_v = max(patch[3] - self.image_sizes[patch[5]][1], 0)
        pad_h = max(patch[2] - self.image_sizes[patch[5]][0], 0)

        mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        if pad_v > 0:
            mask[self.patch_size-int(pad_v)-1:self.patch_size, :] = 255
        if pad_h > 0:
            mask[:, self.patch_size-int(pad_h)-1:self.patch_size] = 255

        for b in bboxes:
            x1, y1, x2, y2, _, _ = self.gt_boxes[b]
            x1 = max(x1-patch[0], 0)
            y1 = max(y1-patch[1], 0)
            x2 = min(x2-patch[0], self.patch_size)
            y2 = min(y2-patch[1], self.patch_size)
            mask[y1:y2, x1:x2] = 127  # to test/visualize
        mask = Image.fromarray(mask, mode='L')

        if self.training is True:
            mask = trsf_hflip(mask)
        mask = torch.LongTensor(np.array(mask).astype(np.uint8))

        print("Image {} with patch grid of {} by {}".format(
            self.image_files[patch[5]],
            self.patch_grids[patch[5]][0],
            self.patch_grids[patch[5]][1]))
        return image, mask  # , patch

    @staticmethod
    def collate_fn(batch):
        images = [item[0] for item in batch]
        gt = [item[1] for item in batch]
        images = torch.stack(images, dim=0)
        gt = torch.stack(gt, dim=0)
        # patches = [item[2] for item in batch]

        return [images, gt]

    def __str__(self):
        return "PatchBasedDataset\npatch_size={}\nImage_count={}\n"\
            "Patches_count={}".format(
                self.patch_size, len(self.image_files), len(self.patches))


def _get_patches_coords(image_sizes, patch_size):
    patches = []
    patch_grids = []
    stride = int(patch_size / 2)
    for k, dims in enumerate(image_sizes):
        width, height = dims

        n_w = math.ceil(width / patch_size) * 2 - 1 \
            - int((width % patch_size) < stride)
        n_h = math.ceil(height / patch_size) * 2 - 1 \
            - int((height % patch_size) < stride)
        n_w = n_w if n_w > 0 else 1
        n_h = n_h if n_h > 0 else 1

        for j in range(n_h):
            for i in range(n_w):
                patches.append([i*stride, j*stride,
                               i*stride+patch_size-1, j*stride+patch_size-1,
                               patch_size*patch_size, k])
        patch_grids.append([n_w, n_h])
    return patches, patch_grids


def _load_image(path, filename, bits, mode):
    if filename.rsplit('.')[1].lower() == 'dcm':
        ds = pydicom.dcmread(os.path.join(path, filename))
        m = ('I;16' if bits == 16 else 'L') if mode == 'L' else 'RGB'
        image = Image.frombuffer(
            m, (ds.Columns, ds.Rows), ds.PixelData, 'raw', m, 0, 1)
    else:
        image = Image.open(os.path.join(path, filename)).convert(mode)
    return image


def _get_image_dimensions(path, image_files, mode='L', bits=8):
    dimensions = []
    for filename in image_files:
        image = _load_image(path, filename, bits, mode)
        width, height = image.size
        dimensions.append([width, height])
    return dimensions


def _associate_bbox_to_patch(patch_coords, gt_boxes,
                             n_images, iou_threshold=0.5):

    patches_per_image = [[_ for _ in patch_coords if _[5] == k]
                         for k in range(n_images)]
    gt_bbox_per_image = [[_ for _ in gt_boxes if _[5] == k]
                         for k in range(n_images)]

    gt_box_to_patch, idx = [], 0
    for patches, gt_bbox in zip(patches_per_image, gt_bbox_per_image):
        if len(gt_bbox) > 0:
            _, _, inter_over_area2 = compute_iou_matrix(patches, gt_bbox)
            bb = [np.nonzero(inter_over_area2[p] > iou_threshold)
                  for p in np.arange(len(patches))]
            bb = [(np.array(elements)+idx).tolist()[0] for elements in bb]
            gt_box_to_patch.append(bb)
            idx += len(gt_bbox)
        else:
            gt_box_to_patch.append([[]]*len(patches))

    all_patches = list(itertools.chain.from_iterable(patches_per_image))
    gt_box_per_patch = list(itertools.chain.from_iterable(gt_box_to_patch))

    print("Filtering with IoU treshold of {}:\n"
          " Patches count={}".format(
              iou_threshold, len(all_patches)))

    return all_patches, gt_box_per_patch,\
        list(itertools.chain.from_iterable(gt_bbox_per_image))


def _filter_positive_samples(patch_coords, gt_boxes,
                             n_images, iou_threshold=0.5):

    patches_per_image = [[_ for _ in patch_coords if _[5] == k]
                         for k in range(n_images)]
    gt_bbox_per_image = [[_ for _ in gt_boxes if _[5] == k]
                         for k in range(n_images)]

    positive_patches = []
    negative_patches = []
    gt_box_to_positive, idx = [], 0
    for patches, gt_bbox in zip(patches_per_image, gt_bbox_per_image):
        if len(gt_bbox) > 0:
            _, _, inter_over_area2 = compute_iou_matrix(patches, gt_bbox)
            pos = np.sum(inter_over_area2 > iou_threshold, axis=1) > 0
            positive_patches.append([a for a, b in zip(patches, pos) if b])
            negative_patches.append([a for a, b in zip(patches, pos) if not b])
            bb = [np.nonzero(inter_over_area2[p] > iou_threshold)
                  for p in np.arange(len(patches))[pos]]
            bb = [(np.array(elements)+idx).tolist()[0] for elements in bb]
            gt_box_to_positive.append(bb)
            idx += len(gt_bbox)
        else:
            negative_patches.append([a for a in patches])

    positive_patches = list(itertools.chain.from_iterable(positive_patches))
    negative_patches = list(itertools.chain.from_iterable(negative_patches))
    gt_box_to_positive = list(
        itertools.chain.from_iterable(gt_box_to_positive))

    print("Filtering with IoU treshold of {}:\nPositive"
          " patches count={}\nNegative patches count={}".format(
              iou_threshold, len(positive_patches), len(negative_patches)))

    return positive_patches, gt_box_to_positive,\
        list(itertools.chain.from_iterable(gt_bbox_per_image))
