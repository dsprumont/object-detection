## Applications

This directory contains scripts that use Detectron2

* `fal_syn_dataset.py`

   This script contains tools to generate train/val/test datasets from original FAL SYN.  
   The parameters are hardcoded in the script (future update might use argparser for cleaner usage.  

   Usage:  
   ```  
   python fal_syn_dataset.py  
   ```  

* `train_backbone.py`

   This script is used to train a [backbone model](modelling/) from scratch with Detectron2 core functions.  
   * Training is actually performed on single GPU.  
   * The script only takes a .yaml as input. The specific yaml node (MODEL.WEIGHTS, OUTPUT_DIR, ..) attribution is not yet available.  
   * Dataset is defined in the main function. Detectron2 DatasetCatalog/MetadataCatalog are used to register the dataset.

   To train the model from a given configuration:  
   ```  
   python train_backbone.py --model-config configs/base_feature_extractor.yaml  
   ```  

   To evaluate a trained model:  
   ```  
   python train_backbone.py --model-config configs/base_feature_extractor.yaml --eval-only  
   ```  
   This evaluate the model over the cfg.DATASET.TEST defined in the config file.  

   For all options, see `python train_backbone.py -h`.