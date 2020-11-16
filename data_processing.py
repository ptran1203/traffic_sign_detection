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


image_feature_description = {
    "bbox": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
}


def preprocess_data(example):
    """
    Applies preprocessing step to a single example
    """
    sample = tf.io.parse_single_example(example, image_feature_description)
    image = tf.image.decode_png(sample["image"])
    bbox = tf.cast(
        tf.io.decode_raw(sample["bbox"], out_type=tf.int64), dtype=tf.float32
    )
    bbox = to_xyxy(tf.reshape(bbox, (-1, 4)))
    bbox = normalize_bbox(bbox)

    # Data augmentation
    image, bbox = augmentation.random_flip_horizontal(image, bbox)
    image = augmentation.random_adjust_brightness(image)
    image = augmentation.random_adjust_contrast(image)
    image = augmentation.random_gaussian_blur(image)

    image, image_shape, _ = resize_and_pad_image(image)
    w, h = image_shape[0], image_shape[1]
    bbox = tf.stack(
        [bbox[:, 0] * h, bbox[:, 1] * w, bbox[:, 2] * h, bbox[:, 3] * w], axis=-1,
    )
    bbox = convert_to_xywh(bbox)
    label = tf.io.decode_raw(sample["label"], out_type=tf.int64)

    return image, bbox, label


def image_bboxes(annotations):
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
    image_shape = tf.image.decode_jpeg(image_string).shape

    feature = {
        "bbox": bytes_feature(bbox),
        "label": bytes_feature(label),
        "image": bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecords(data, file_path, train_dir):
    count = 0
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

