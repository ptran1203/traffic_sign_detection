import tensorflow as tf
import numpy as np
import data_augmentation as augmentation
from utils import (
    resize_and_pad_image,
    swap_xy,
    convert_to_xywh,
    convert_to_corners,
    to_xyxy,
    normalize_bbox,
)
import math
import os

image_feature_description = {
    "bbox": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
}

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_array_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def has_small_bbox(bboxes):
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    min_val = tf.constant(650, dtype=tf.float32)
    return tf.math.reduce_any(tf.math.less(areas, min_val))

def create_dataset_list(annotations):
    image_bboxes = {}
    for item in annotations:
        img_id = item.get("image_id")
        if img_id in image_bboxes:
            image_bboxes[img_id]["bbox"].append(item["bbox"])
            image_bboxes[img_id]["label"].append(item["category_id"])
        else:
            image_bboxes[img_id] = {
                "id": img_id,
                "bbox": [item["bbox"]],
                "label": [item["category_id"]],
            }

    return list(image_bboxes.values())


def image_example(image_string, label, bbox):
    feature = {
        "bbox": bytes_feature(bbox),
        "label": bytes_feature(label),
        "image": bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecords(data, file_path, train_dir):
    count = 0
    if train_dir.endswith("images"):
        train_dir = train_dir.replace("images", "")

    with tf.io.TFRecordWriter(file_path) as writer:
        for img_info in data:
            ipath = "{}images/{}.png".format(train_dir, img_info["id"])
            image_string = open(ipath, "rb").read()
            tf_example = image_example(
                image_string,
                np.array(img_info["label"]).tobytes(),
                np.array(img_info["bbox"]).tobytes(),
            )
            writer.write(tf_example.SerializeToString())
            count += 1
            if count % 100 == 0:
                print(count, "/", len(data))

class DataProcessing:
    def __init__(self, origin_width=1622, origin_height=626 , width=400,
                height=154, augment=True, mix_iterator=None,convert_xywh=True,
                random_cropping=True, dynamic_size=False):
        self.origin_width = origin_width
        self.origin_height = origin_height
        self.dynamic_size = dynamic_size
        self.width = width
        self.height = height
        self.random_cropping = random_cropping
        self.scale_x = self.origin_width / self.width
        self.scale_y = self.origin_height / self.height
        self.convert_xywh = convert_xywh
        self.augment = augment
        self.mix_iterator = mix_iterator

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

    def moved_box(self, box, x1, y1, width, height):
        x1, y1 = tf.cast(x1, tf.float32), tf.cast(y1, tf.float32)

        return tf.stack([
            (box[:, 0] - x1) * self.scale_x,
            (box[:, 1] - y1) * self.scale_y,
            (box[:, 2] - x1) * self.scale_x,
            (box[:, 3] - y1) * self.scale_y,
        ], axis=1)


    def get_slice_indices(self):
        num_paths = math.ceil(self.origin_width / self.width)
        slices = []
        for i in range(num_paths):
            start = max(self.width * i - self.overlap_x, 0)
            end = start + self.width
            if end > self.origin_width:
                start = end - self.origin_width
                end = self.origin_width

            slices.append([start, end])

        return slices

    def random_crop(self, image, bbox, labels):
        width = self.width
        height = self.height
        idx = tf.random.uniform((), 0, tf.shape(bbox)[0], tf.int32)
        selected_box = bbox[idx]
        x1, y1, x2, y2 = tf.unstack(selected_box, axis=0)
        x1 = tf.cast(x1, tf.int32)
        x2 = tf.cast(x2, tf.int32)
        y1 = tf.cast(y1, tf.int32)
        y2 = tf.cast(y2, tf.int32)

        # 60% part of object lie inside the frame is considered valid
        accept_ratio = 0.6
        mean_x1, mean_x2 = tf.reduce_mean(bbox[:, 0]), tf.reduce_mean(bbox[:, 2])
        pad_size = accept_ratio * (mean_x2 - mean_x1)

        x1 = tf.random.uniform((), x1 - width, x1, dtype=tf.int32)
        y1 = tf.random.uniform((), y1 - height, y1, dtype=tf.int32)

        if tf.less(x1, 0):
            x1 = 0

        if tf.less(y1, 0):
            y1 = 0

        if tf.greater(x1 + width, self.origin_width):
            x1 = self.origin_width - width

        if tf.greater(y1 + height, self.origin_height):
            y1 = self.origin_height - height

        if tf.greater(y2, y1 + height):
            y1 = y1 + (y2 - (y1 + height))

        if tf.greater(x2, x1 + width):
            x1 = x1 + (x2 - (x1 + width))

        # [height, width, channels]
        cropped = tf.slice(image, [y1, x1, 0], [height, width, 3])

        x1 = tf.cast(x1, tf.float32)
        y1 = tf.cast(y1, tf.float32)
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        # filter out boxes that not lie inside the cropped image
        x1_b, y1_b, x2_b, y2_b = tf.unstack(bbox, axis=1)

        # 1. x1 of box > cropped width
        # 2. x2 of box < cropped width
        x_condition = tf.logical_and(
            tf.greater(x1_b, x1 - pad_size),
            tf.less(x2_b, x1 + width + pad_size)
        )
        # 3. y1 of box> cropped height
        # 4. y2 of box> cropped height
        y_condition = tf.logical_and(
            tf.greater(y1_b, y1 - pad_size),
            tf.less(y2_b, y1 + height + pad_size)
        )

        cond = tf.logical_and(x_condition, y_condition)
        positive_mask = tf.where(cond)

        bbox = self.moved_box(bbox, x1, y1, width, height)
        bbox = tf.gather_nd(bbox, positive_mask)
        labels = tf.gather_nd(labels, positive_mask)

        return cropped, bbox, labels

    def preprocess_data(self, example):
        """
        Applies preprocessing step to a single example
        """
        sample = tf.io.parse_single_example(example, image_feature_description)
        image = tf.image.decode_png(sample["image"])
        bbox = tf.cast(
            tf.io.decode_raw(sample["bbox"], out_type=tf.int64), dtype=tf.float32
        )

        label = tf.io.decode_raw(sample["label"], out_type=tf.int64)
        bbox = to_xyxy(tf.reshape(bbox, (-1, 4)))

        if self.dynamic_size:
            shape = tf.shape(image)
            self.origin_width = shape[1]
            self.origin_height = shape[0]

        if not self.augment:
            image, bbox, label = self.random_crop(image, bbox, label)
            image = tf.image.resize(image, (self.origin_height, self.origin_width))
            if self.convert_xywh:
                bbox = convert_to_xywh(bbox)
            return image, bbox, label

        # Data augmentation
        image = augmentation.random_adjust_brightness(image)
        image = augmentation.random_adjust_contrast(image)
        # crop the region contain at least 1 bounding box
        has_smallb = has_small_bbox(bbox)
        if self.random_cropping and tf.logical_or(has_smallb, tf.random.uniform(()) > 0.5):
            image, bbox, label = self.random_crop(image, bbox, label)

        bbox = normalize_bbox(bbox, self.origin_width, self.origin_height)
        image, bbox = augmentation.random_flip_horizontal(image, bbox)

        if not has_smallb:
            image = augmentation.random_gaussian_blur(image, 0.5)

        image, image_shape, _ = resize_and_pad_image(image, jitter=None)
        w, h = image_shape[0], image_shape[1]

        bbox = tf.stack([
            bbox[:, 0] * h,
            bbox[:, 1] * w,
            bbox[:, 2] * h,
            bbox[:, 3] * w,
        ], axis=-1)

        if self.convert_xywh:
            bbox = convert_to_xywh(bbox)

        return image, bbox, label
