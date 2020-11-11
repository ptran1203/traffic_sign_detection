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


image_feature_description = {
    "bbox": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "image": tf.io.FixedLenFeature([], tf.string),
}


def image_bboxes(annotations):
    image_bboxes = {}
    for item in annotations:
        img_id = item.get("image_id")
        if img_id in image_bboxes:
            image_bboxes[img_id]["bbox"].append(item["bbox"])
        else:
            image_bboxes[img_id] = {
                "id": img_id,
                "bbox": [item["bbox"]],
                "label": item["category_id"],
            }

    return list(image_bboxes.values())


def preprocess_data(example):
    """
    Applies preprocessing step to a single example
    """
    sample = tf.io.parse_single_example(example, image_feature_description)
    image = tf.image.decode_image(sample["image"])
    bbox = tf.io.decode_raw(sample["bbox"], out_type=tf.int64)
    label = sample["label"]

    return image, bbox, label


def image_example(image_string, label, bbox):
    image_shape = tf.image.decode_jpeg(image_string).shape

    feature = {
        "bbox": bytes_feature(bbox),
        "label": int64_feature(label),
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
                image_string, img_info["label"], np.array(img_info["bbox"]).tobytes()
            )
            writer.write(tf_example.SerializeToString())
            count += 1
            if count % 100 == 0:
                print(count, "/", len(data))
