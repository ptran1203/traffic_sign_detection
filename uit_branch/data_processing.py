import tensorflow as tf
import numpy as np
from data_augmentation import (
    random_adjust_brightness,
    random_adjust_contrast,
    random_flip_horizontal,
    random_gaussian_blur
)
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

def has_small_bbox(width, height, bboxes):
    w, h = (bboxes[:, 2] - bboxes[:, 0]), (bboxes[:, 3] - bboxes[:, 1])
    rate_w, rate_h = w / width, h / height
    return tf.math.reduce_any(
        tf.logical_or(
            tf.math.less(rate_w, 0.08),
            tf.math.less(rate_h, 0.08)
        )
    )

class DataProcessing:
    """
    Some function are implemented at traffic_sign_detection/data_processing.py
    """
    def __init__(self,
                iterator=None,
                is_iter=False,
                convert=True,
                resize=512,
                augment=True,
                crop_width=256,
                crop_height=256):
        self.convert = convert
        self.iterator = iterator
        self.is_iter = is_iter
        self.resize = resize
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.augment = augment

    def moved_box(self, box, x1, y1, scale_x, scale_y):
        x1, y1 = tf.cast(x1, tf.float32), tf.cast(y1, tf.float32)

        return tf.stack([
            (box[:, 0] - x1) * scale_x,
            (box[:, 1] - y1) * scale_y,
            (box[:, 2] - x1) * scale_x,
            (box[:, 3] - y1) * scale_y,
        ], axis=1)

    def set_height(self, height):
        self.origin_height = height
        self.crop_height = height // 3

    def set_width(self, width):
        self.origin_width = width
        self.crop_width = width // 3

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
        width = self.crop_width
        height = self.crop_height
        idx = tf.random.uniform((), 0, tf.shape(bbox)[0], tf.int32)
        selected_box = bbox[idx]
        x1, y1, x2, y2 = tf.unstack(selected_box, axis=0)
        x1 = tf.cast(x1, tf.int32)
        x2 = tf.cast(x2, tf.int32)
        y1 = tf.cast(y1, tf.int32)
        y2 = tf.cast(y2, tf.int32)

        pad_size = 10

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

        bbox = self.moved_box(
            bbox,
            x1,
            y1,
            tf.cast(self.origin_width, tf.float32) / width,
            tf.cast(self.origin_height, tf.float32) / height)

        bbox = tf.gather_nd(bbox, positive_mask)
        labels = tf.gather_nd(labels, positive_mask)

        return cropped, bbox, labels

    def decode_sample(self, example):
        sample = tf.io.parse_single_example(example, image_feature_description)
        image = tf.image.decode_png(sample["image"])
        bbox = tf.cast(
            tf.io.decode_raw(sample["bbox"], out_type=tf.int64), dtype=tf.float32
        )
        label = tf.io.decode_raw(sample["label"], out_type=tf.int64)
        bbox = tf.reshape(bbox, (-1, 4))

        shape = tf.shape(image)


        self.set_height(shape[0])
        self.set_width(shape[1])

        shape = tf.cast(shape, tf.float32)
        width = shape[1]
        height = shape[0]

        bbox = tf.stack([
            tf.maximum(bbox[:, 0], 0),
            tf.maximum(bbox[:, 1], 0),
            tf.minimum(bbox[:, 2], width),
            tf.minimum(bbox[:, 3], height), 
        ], axis=-1)

        
        if has_small_bbox(width, height, bbox) and tf.random.uniform(()) > 0.5:
            image, bbox, label = self.random_crop(image, bbox, label)
        
        if self.augment:
            image = random_adjust_brightness(image)
            image = random_adjust_contrast(image)

            if tf.random.uniform(()) >= 0.8:
                image = tf.image.random_hue(image, 0.1)

            if tf.random.uniform(()) >= 0.8:
                image = tf.image.random_saturation(image, 0.1, 0.5)

        bbox = normalize_bbox(bbox, width, height)

        image, bbox = random_flip_horizontal(image, bbox, 0.5)
        image, image_shape, _ = resize_and_pad_image(image,
                                                     self.resize,
                                                     self.resize, jitter=None)
        w, h = image_shape[0], image_shape[1]

        bbox = tf.stack([
            bbox[:, 0] * h,
            bbox[:, 1] * w,
            bbox[:, 2] * h,
            bbox[:, 3] * w,
        ], axis=-1)

        if self.iterator and not self.is_iter and tf.random.uniform(()) > 0.5:
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
            # image, bbox, label = self.copy_paste(
            #     image, bbox, label,
            #     image_, bbox_, label_
            # )

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