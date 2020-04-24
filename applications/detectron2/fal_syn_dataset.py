import os
import shutil
import random
import numpy as np
import cv2
import json

from detectron2.structures import BoxMode

"""
    Setup directories and {train, val, test} splits according to a given configuration.

    Also create a meta_info.json file un each subdirectories contains information
    relevant to detectron2 input format (to be read by get_dataset(path), get_cls_dataset(path)).
"""
def generate_datasets(data_path, config):

    # Several assertions 
    assert(config["train"] > 0)
    assert(config["val"] > 0)
    assert(config["test"] >=0)
    norm = config["train"] + config["val"] + config["test"]

    # Create {train,val[,test]} directories 
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')
    test_dir = os.path.join(data_path, 'test')

    if os.path.exists(train_dir): shutil.rmtree(train_dir)
    if os.path.exists(val_dir): shutil.rmtree(val_dir)
    if os.path.exists(test_dir): shutil.rmtree(test_dir)

    os.makedirs(train_dir)
    os.makedirs(val_dir)
    if config["test"] > 0: 
        os.makedirs(test_dir)

    # Check if every label file has a corresponding image
    images_dir = os.path.join(data_path, 'images')
    labels_dir = os.path.join(data_path, 'labels')

    label_files = os.listdir(labels_dir)
    for label_file in label_files:
        file_name = str(label_file).rsplit(".txt")[0]
        assert(os.path.exists(os.path.join(images_dir, file_name+'.jpg')))
    
    # Shuffle the labels list
    random.seed(config["seed"])
    random.shuffle(label_files)

    # Get the splitting values
    n_total = len(label_files)
    n_train = int(np.floor(config["train"] * n_total / norm))
    n_val = int(np.floor(config["val"] * n_total / norm) + n_train)

    # Loop over every labels and store them into the according directory, 
    # along with a json file containing annotations {bbox, segmentation, category} 
    # for each image
    dataset_dicts = []
    split_offset = 0
    output_dir = os.path.join(data_path, 'train')
    for n, label_file in enumerate(label_files):
        
        if n == n_train:
            with open(os.path.join(train_dir, 'meta_info.json'), 'w') as f:
                info = { "name": "train_dataset", "data": dataset_dicts }
                json.dump(info, f, indent=4, separators=(',', ': '))

            dataset_dicts = []
            split_offset = n_train
            output_dir = os.path.join(data_path, 'val')
        elif n == n_val:
            with open(os.path.join(val_dir, 'meta_info.json'), 'w') as f:
                info = { "name": "val_dataset", "data": dataset_dicts }
                json.dump(info, f, indent=4, separators=(',', ': '))

            dataset_dicts = []
            split_offset = n_val
            output_dir = os.path.join(data_path, 'test')
        else:
            pass

        img_name = str(label_file).replace('.txt', '.jpg')
        shutil.copy(os.path.join(images_dir, img_name),os.path.join(output_dir, img_name))

        record = {}
        objs = []

        filename = os.path.join(output_dir, img_name)
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = n-split_offset
        record["height"] = height
        record["width"] = width

        with open(os.path.join(labels_dir, label_file)) as f:
            for line in f.readlines():
                words = line.split(' ')
                assert(len(words) == 5)
                category = words[0]
                xmin = int(words[1])
                xmax = int(words[2])
                ymin = int(words[3])
                ymax = int(words[4])

                px = [xmin, xmax, xmax, xmin, xmin]
                py = [ymin, ymin, ymax, ymax, ymin]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                obj = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": int(config["mapping"][category]),
                "iscrowd": 0
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    if config["test"] > 0:
        with open(os.path.join(test_dir, 'meta_info.json'), 'w') as f:
            info = { "name": "test_dataset", "data": dataset_dicts }
            json.dump(info, f, indent=4, separators=(',', ': '))
    else:
        with open(os.path.join(val_dir, 'meta_info.json'), 'w') as f:
            info = { "name": "val_dataset", "data": dataset_dicts }
            json.dump(info, f, indent=4, separators=(',', ': '))


def get_dataset(dataset_dir):
    with open(os.path.join(dataset_dir, 'meta_info.json')) as f:
        meta_info = json.load(f)
        for element in meta_info['data']:
          for anno in element['annotations']:
            # it seems that because we save the dict in a json file,
            # the <BoxMode.XYXY_ABS: 0> object gets convert to (int) 0.
            # Later on when converting to coco, it calls the dict method .value()
            # on a int, which raise an error
            anno['bbox_mode'] = BoxMode.XYXY_ABS
        return meta_info['data']


def get_cls_dataset(dataset_dir):
    with open(os.path.join(dataset_dir, 'meta_info.json')) as f:
        meta_info = json.load(f)

        dict_list = []
        image_dict = {}
        for element in meta_info['data']:
            image_dict = {
                "file_name": element["file_name"],
                "image_id": element["image_id"],
                "height": element["height"],
                "width": element["width"],
                "label": 0 # intact
            }
            if len(element["annotations"]) > 0:
                image_dict["label"] = 1 # defective
                
            dict_list.append(image_dict)
        return dict_list


if __name__ == '__main__':

    # PATH to dataset
    path = 'C:/Users/sprum/Workspace/Anaconda3/TFE/defect-detection/datasets/ADRIC-XRIS-FAL-SYN-SIMP'

    # CONFIG 
    config = {
        "seed": 1234,       # seed to control randomness in split generation
        "train": 0.7,       # percentage of images in training dataset (must be >0)
        "val": 0.1,         # percentage of images in validation dataset (must be >0)
        "test": 0.2,        # percentage of images in test dataset (can be 0)
        "mapping": {        # mapping from cytomine label to detectron label
            "42459392": 0,  # here 2 kinds of annotation are merged into one
            "42459398": 0
        }
    }

    generate_datasets(path, config)