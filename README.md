# Small Object Detection
Implementation of Automated Small Object Detection System with Deep Learning.
The objectives of this project are

 1. to evaluate several deep learning architectures over small object-based datasets
 2. to integrate the detection pipeline with [Cytomine](https://cytomine.coop/), an open-source web platform which fosters collaborative analysis of very large images and allows semi-automatic processing of large image collections via machine learning algorithms.

The works is currently heavily based on [Detectron2](https://github.com/facebookresearch/detectron2), a deep learning-based detection framework developed by [Facebook](https://ai.facebook.com/research/), and powered by Pytorch.

## Applications

See [Applications](applications/detectron2/) for a list of applications built with Detectron2.

- [x] Train a simple backbone model (Resnet-like)
- [ ] Train a full detector model (Faster-rcnn, ..)
- [ ] Fetch data from cytomine directly into detectron2 
- [ ] .. 

## Licence

This project is released under the [Apache 2.0 license](LICENSE).


