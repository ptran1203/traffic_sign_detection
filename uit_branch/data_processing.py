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

class Decoder:
    def __init__(self,
                iterator=None,
                is_iter=False,
                convert=True,
                crop_size=512,
                augment=True):
        self.convert = convert
        self.iterator = iterator
        self.is_iter = is_iter
        self.crop_size = crop_size
        self.augment = augment

    def decode_sample(self, example):
        sample = tf.io.parse_single_example(example, image_feature_description)
        image = tf.image.decode_png(sample["image"])
        bbox = tf.cast(
            tf.io.decode_raw(sample["bbox"], out_type=tf.int64), dtype=tf.float32
        )
        label = tf.io.decode_raw(sample["label"], out_type=tf.int64)
        bbox = tf.reshape(bbox, (-1, 4))

        shape = tf.cast(tf.shape(image), tf.float32)

        bbox = tf.stack([
            tf.maximum(bbox[:, 0], 1),
            tf.maximum(bbox[:, 1], 1),
            tf.minimum(bbox[:, 2], shape[1] - 1),
            tf.minimum(bbox[:, 3], shape[0] - 1),       
        ], axis=-1)
        
        bbox = normalize_bbox(bbox,
                              tf.cast(shape[1], tf.float32),
                              tf.cast(shape[0], tf.float32))
        
        if self.augment:
            image, bbox = random_flip_horizontal(image, bbox, 0.5)
            image = random_adjust_brightness(image)
            image = random_adjust_contrast(image)

            if tf.random.uniform(()) > 0.8:
                image = tf.image.random_hue(image, 0.1)

            if tf.random.uniform(()) > 0.8:
                image = tf.image.random_saturation(image, 0.1, 0.5)

        image, image_shape, _ = resize_and_pad_image(image,
                                                     self.crop_size,
                                                     self.crop_size, jitter=None)
        w, h = image_shape[0], image_shape[1]

        bbox = tf.stack([
            bbox[:, 0] * h,
            bbox[:, 1] * w,
            bbox[:, 2] * h,
            bbox[:, 3] * w,
        ], axis=-1)

        if self.iterator and not self.is_iter:
            image_, bbox_, label_ = self.iterator.get_next()
            shape = tf.shape(image)
            shape_ = tf.shape(image_)

            # mixup
            if shape_[0] == shape[0] and shape[1] == shape_[1]:
                image = tf.cast(image, tf.float32)
                if tf.size(label_) > 0:
                    bbox = tf.concat([bbox, bbox_], axis=0)
                    label = tf.concat([label, label_], axis=0)
                    r = tf.random.uniform((), 0.35, 0.65)
                    image = image * r + image_ * (1 - r)

            # copy-paste
            image, bbox, label = self.copy_paste(
                image, bbox, label,
                image_, bbox_, label_
            )

        if self.convert and not self.is_iter:
            bbox = convert_to_xywh(bbox)

        return image, bbox, label

    def copy_paste(self, image_1, box_1, label_1, image_2, box_2, label_2):
        """Copy objects from image_2 to image_1"""
        return  image_1, box_1, label_1


    def mixup(self, img1, img2, mixup_ratio):
        # TODO: perform slice assignment
        mix_img[:shape1[0], :shape1[1], :].assign()
        mix_img[:shape2[0], :shape2[0], :].assign(mix_img[:shape2[0], :shape2[0], :] +  img2 * (1-mixup_ratio))
        return mix_img