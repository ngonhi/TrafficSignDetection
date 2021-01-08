""" Sample TensorFlow CSV-to-TFRecord converter

usage: generate_tfrecord.py [-h] [--images IMAGE_DIR] [--annot CSV_PATH] [--label_map LABEL_MAP_PATH] [--tfrecords TFRECORDS_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --images IMAGE_DIR
                        Path to the folder where the input image files are stored
                        Default: data/train/images
  --annot CSV_PATH
                        Absolute path to csv file
                        Default: annotations.csv
  --label_map LABEL_MAP_PATH
                        Absolute path to labelmap file
                        Default: label_map.pbtxt
  --tfrecords TFRECORDS_PATH
                        Absolute path to where to save the TFrecord file
                        Default: data/myrecord.record

CSV file has to be in format
filename, width, height, class, xmin, ymin, xmax, ymax

Remember to git clone https://github.com/tensorflow/models.git

Ref: https://github.com/abdelrahman-gaber/tf2-object-detection-api-tutorial
"""

import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow.compat.v1 as tf
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

# Initiate argument parser
parser = argparse.ArgumentParser(description="Generating tfrecords from images and csv file")
parser.add_argument("--images", type=str, help="folder that contains images",
                    default="data/train/images")
parser.add_argument("--annot", type=str, help="full path to annotations csv file",
                    default="annotations.csv")
parser.add_argument("--label_map", type=str, help="full path to label_map file",
                    default="label_map.pbtxt")
parser.add_argument("--save_tfrecords", type=str, help="This path is for saving the generated tfrecords",
                    default="data/myrecord.record")
args = parser.parse_args()


label_map = label_map_util.load_labelmap(args.label_map)
label_map_dict = label_map_util.get_label_map_dict(label_map)

WIDTH = 1622
HEIGHT = 626


def class_text_to_int(row_label):
    return label_map_dict[row_label]


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()

    width = WIDTH
    height = HEIGHT
    filename = group.filename.encode('utf8')
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    csv_path = args.annot
    images_path = args.images
    print("images path : ", images_path)
    print("csv path : ", csv_path)
    print("path to output tfrecords : ", args.tfrecords)
    label_map_dict = label_map_util.get_label_map_dict(args.label_map)
    writer = tf.io.TFRecordWriter(args.tfrecords)

    examples = pd.read_csv(csv_path)
    print("Generating tfrecord .... ")
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, images_path, label_map_dict)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(args.tfrecords))


if __name__ == '__main__':
    tf.app.run()
