import tensorflow as tf
import numpy as np


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

# Internal
_H, _W = 626, 1622
H, W = 512, 1024
ratio_w = _W / W
offset_h = (_H - H) // 2


def preprocess_data(example):
    """
    Applies preprocessing step to a single example
    """
    sample = tf.io.parse_single_example(example, image_feature_description)
    image = tf.image.decode_png(sample["image"])
    image = tf.image.resize_with_pad(image, H, W)
    bbox = tf.cast(
        tf.io.decode_raw(sample["bbox"], out_type=tf.int64), dtype=tf.float32
    )
    bbox = tf.reshape(bbox, (-1, 4))
    bbox = tf.stack(
        [
            (bbox[:, 0] / ratio_w),
            (bbox[:, 1] / ratio_w) + offset_h,
            (bbox[:, 2] / ratio_w),
            (bbox[:, 3] / ratio_w),
        ],
        axis=-1,
    )
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
                "label": [item["category_id"],],
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
