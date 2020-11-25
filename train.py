import json
import pandas as pd
import numpy as np
import tensorflow as tf
import model as m
import data_processing
import losses
import utils
import datetime
import os
import argparse

parser = argparse.ArgumentParser(description='Traffic sign detection')
parser.add_argument("--input", dest="input_path",
                    metavar="I", type=str, default="/data/images",
                    help="Path to training images")
parser.add_argument("--batch", dest="batch_size",
                    metavar="B", type=int, default=2)
args = parser.parse_args()

input_path = args.input_path

TFRECORDS_FILE = "./images.tfrecords"
WEIGHT_FILE = "weight_dense.h5"
metadata = json.load(open("./train_traffic_sign_dataset.json", "r"))

print("Create tfrecords dataset")
data_processing.write_tfrecords(
    data_processing.create_dataset_list(metadata["annotations"]),
    TFRECORDS_FILE,
    input_path
)

autotune = tf.data.experimental.AUTOTUNE

label_map = {
    1: "No entry",
    2: "No parking / waiting",
    3: "No turning",
    4: "Max Speed",
    5: "Other prohibition signs",
    6: "Warning",
    7: "Mandatory",
}
num_classes = 7
batch_size = args.batch_size


fdataset = tf.data.TFRecordDataset(TFRECORDS_FILE)
data_processor = data_processing.DataProcessing(400, 154)
label_encoder = m.LabelEncoder()
dataset = fdataset.map(data_processor.preprocess_data)
dataset = dataset.shuffle(8 * batch_size)
dataset = dataset.padded_batch(
    batch_size,
    padding_values=(0.0, 1e-8, tf.cast(-1, tf.int64)),
    drop_remainder=True,
)
dataset = dataset.map(
    label_encoder.encode_batch, num_parallel_calls=autotune
)
dataset = dataset.apply(tf.data.experimental.ignore_errors())
dataset = dataset.prefetch(autotune)

val_size = 500
train_size = 4500 - val_size
train_data = dataset.take(train_size)
val_data = dataset.skip(train_size) 
train_steps_per_epoch = train_size // batch_size
train_steps = 6 * 10000
epochs = train_steps // train_steps_per_epoch

learning_rates = [1e-4, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)
optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=WEIGHT_FILE,
        monitor="loss",
        save_best_only=False,
        save_weights_only=True,
        verbose=1,
    )
]

backbone = "densenet121"

model = m.RetinaNet(num_classes, backbone=backbone)
model.compile(optimizer=optimizer, loss=losses.RetinaNetLoss(num_classes))
model.fit(np.random.rand(1, 896, 2304, 3), np.random.rand(1, 386694, 5))
utils.try_ignore_error(model.load_weights, WEIGHT_FILE)

H = model.fit(train_data.repeat(),
              validation_data=val_data,
              epochs=epochs,
              steps_per_epoch=train_steps_per_epoch,
              callbacks=callbacks_list)
