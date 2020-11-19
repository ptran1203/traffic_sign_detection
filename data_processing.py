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

def has_small_bbox(bboxes):
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    min_val = tf.constant(650, dtype=tf.float32)
    return tf.math.reduce_any(tf.math.less(areas, min_val))


def moved_box(box, x1, x2, y1, y2):
    x1, x2 = tf.cast(x1, tf.float32), tf.cast(x2, tf.float32)
    y1, y2 = tf.cast(y1, tf.float32), tf.cast(y2, tf.float32)
    scale = 4.05500
    return tf.stack([
        (box[:, 0] - x1) * scale,
        (box[:, 1] - y1) * scale,
        (box[:, 2] - x1) * scale,
        (box[:, 3] - y1) * scale,
    ], axis=1)

def random_crop(image, bbox):
    idx = tf.random.uniform((), 0, tf.shape(bbox)[0], tf.int32)
    selected_box = bbox[idx]
    x1, y1, x2, y2 = tf.unstack(selected_box, axis=0)
    x1 = tf.cast(x1, tf.int32)
    x2 = tf.cast(x2, tf.int32)
    y1 = tf.cast(y1, tf.int32)
    y2 = tf.cast(y2, tf.int32)
    w, h = x2 - x1, y2 - y1
    width = 400
    heigh = 154

    x1 = tf.random.uniform((), x1 - width, x1, dtype=tf.int32)
    y1 = tf.random.uniform((), y1 - heigh, y1, dtype=tf.int32)

    if tf.less(x1, 0):
        x1 = 0

    if tf.less(y1, 0):
        y1 = 0

    if tf.greater(x1 + width, 1622):
        x1 = 1622 - width

    if tf.greater(y1 + heigh, 626):
        y1 = 626 - heigh

    if tf.greater(y2, y1 + heigh):
        y1 = y1 + (y2 - (y1 + heigh))

    if tf.greater(x2, x1 + width):
        x1 = x1 + (x2 - (x1 + width))

    croped = tf.slice(image, [y1, x1, 0], [heigh, width, 3])

    x1 = tf.cast(x1, tf.float32)
    y1 = tf.cast(y1, tf.float32)
    width = tf.cast(width, tf.float32)
    heigh = tf.cast(heigh, tf.float32)

    # filter out boxes that not lie inside the cropped image
    x1_b, y1_b, x2_b, y2_b = tf.unstack(bbox, axis=1)
    # 1. x1 of box > cropped width
    # 2. x2 of box < cropped width
    case1 = tf.logical_or(tf.less(x1_b, x1), tf.greater(x2_b, x1 + width + 10))
    # 3. y1 of box> cropped height
    # 4. y2 of box> cropped height
    case2 = tf.logical_or(tf.less(y1_b, y1), tf.greater(y2_b, y1 + heigh + 10))
    cond = tf.logical_or(case1, case2)
    filter = tf.where(tf.logical_not(cond))[0]

    bbox = moved_box(bbox, x1, x2, y1, y2)
    return croped, tf.gather(bbox, filter)

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

    # Data augmentation
    image = augmentation.random_adjust_brightness(image)
    image = augmentation.random_adjust_contrast(image)
    # crop the region contain at least 1 bounding box
    if tf.random.uniform(()) > 0.2:
        image, bbox = random_crop(image, bbox)

    bbox = normalize_bbox(bbox)
    image, bbox = augmentation.random_flip_horizontal(image, bbox)

    image, image_shape, _ = resize_and_pad_image(image, jitter=None)
    w, h = image_shape[0], image_shape[1]
    bbox = tf.stack([
        bbox[:, 0] * h,
        bbox[:, 1] * w,
        bbox[:, 2] * h,
        bbox[:, 3] * w,
    ], axis=-1)
    # bbox = convert_to_xywh(bbox)
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
