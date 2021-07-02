from genericpath import exists
import json
import tensorflow as tf
import model as m
import data_processing
import losses
import utils
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Traffic sign detection')
    parser.add_argument("--input", dest="input_path",
                        metavar="I", type=str, default="/data/images",
                        help="Path to training images")
    parser.add_argument("--backbone", type=str, default='resnet50')
    parser.add_argument("--init-from", type=str, default='resnet50',
                        help='Path to pretrained weight or backbone name')
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-classes", type=int, default=7)
    parser.add_argument("--checkpoint-dir", type=str, default='weights')
    parser.add_argument("--force-tfrec", action='store_true')
    parser.add_argument("--debug-samples", type=int, default=0)

    return parser.parse_args()

def main(args):
    TFRECORDS_FILE = "/tmp/images.tfrecords"
    metadata = json.load(open("./train_traffic_sign_dataset.json", "r"))
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.force_tfrec or not os.path.isfile(TFRECORDS_FILE):
        print("Create tfrecords dataset")
        data_processing.write_tfrecords(
            data_processing.create_dataset_list(metadata["annotations"]),
            TFRECORDS_FILE,
             args.input_path
        )

    autotune = tf.data.experimental.AUTOTUNE
    batch_size = args.batch_size


    fdataset = tf.data.TFRecordDataset(TFRECORDS_FILE)
    data_processor = data_processing.DataProcessing(width=400, height=154)
    label_encoder = m.LabelEncoder()
    dataset = fdataset.map(data_processor.preprocess_data)
    dataset = dataset.shuffle(batch_size)
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

    train_size = args.debug_samples or 4500
    train_data = dataset
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
            filepath=os.path.join(args.checkpoint_dir, f'weight_{args.backbone}.h5'),
            monitor="loss",
            save_best_only=False,
            save_weights_only=True,
            verbose=1,
        )
    ]

    model = m.RetinaNet(args.n_classes, backbone=args.backbone)
    model.compile(optimizer=optimizer, loss=losses.RetinaNetLoss(args.n_classes))
    model.build((1, None, None, 3))
    utils.try_ignore_error(model.load_weights, args.init_from)

    H = model.fit(train_data.repeat(),
                epochs=epochs,
                steps_per_epoch=train_steps_per_epoch,
                callbacks=callbacks_list)


if __name__ == '__main__':
    main(parse_args())