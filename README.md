# TrafficSignDetection
## Overview
This project aims to evaluate multiple object detection models in detecting traffic sign. The dataset we use is taken from [Zalo AI Challenge - Traffic Sign Detection](https://challenge.zalo.ai/portal/traffic-sign-detection). 

## Models
The models used in this project are trained on the following pretrained models:
* EfficientDet D1 ([Tensorflow object detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md))
* SSD MobileNet 320x320 ([Tensorflow object detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md))
* SSD ResNet50 FPN 640x640 ([Tensorflow object detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md))
* Faster R-CNN  ([Tensorflow object detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md))
* YOLOv5 small ([Ultralytics](https://github.com/ultralytics/yolov5))
* YOLOv5 medium ([Ultralytics](https://github.com/ultralytics/yolov5))
* YOLOv5 large ([Ultralytics](https://github.com/ultralytics/yolov5))

## Dataset
In the dataset, we remove groundtruth boxes with area smaller than 40 pixel square, and groundtruth boxes overlapping each other with IoU from 0.7 that annotated the same traffic sign. We annotate the public test set ourselves.

## Results
The following table sums up the results for our models:
|Model|mAP@[.5:.95] validation|mAP@[.5:.95] public test|
|-|-|-|
|YOLO v5 small| 39.3| 35.8|
|YOLO v5 medium| 40.6| 38|
|YOLO v5 large |40.5 |37.5|
|SSD MobileNet |10.7 |7.77|
|SSD ResNet50 FPN |46.7| 20.8|
|EfficientDet D1 |13.2| 8.2|
|Faster R-CNN Resnet50 |26.9 |14.5|

![demo](./demo.gif)

## Train and evaluation
Please review our notebooks in `models` for detailed instructions on how to train and evaluate the models.

## Team members:
- Đặng Chí Trung
- Lâm Văn Sang Em
- Ngô Phương Nhi

## Repository structure
* `models`: contains Jupyter notebook file on how to train and evaluate the models
* `data`: contains 
    *  csv annotation files for training set, validation set and test set, 
    *  `label_map.pbtxt` for TF2 object detection API
    *  exploratory data analysis result
    *  `generate_tfrecord.py`: code to generate tfrecord for TF2 object detection API
    *  `TSD_ReduceData.ipynb`: removing boxes with area smaller than 40
    *  `TSD_SuppressBox.ipynb`: suppress boxes with IoU overlapping from 0.7 that annotate the same traffic sign
* `saved_models`: contains saved models or weights for models.
* `results`: contains some inference results



