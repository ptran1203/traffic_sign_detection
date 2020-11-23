import math
import tensorflow as tf
import utils
import data_processing
import os
import model as m
import losses
import numpy as np
import json
try:
    from google.colab.patches import cv2_imshow
except ImportError:
    try:
        import cv2.imshow as cv2_imshow
    except ImportError:
        def cv2_imshow(img):
            return

class Prediction:
    def __init__(self, inference_model, crop_size=200,
                 crop_height=300, overlap=75):
        self.crop_size = crop_size
        self.crop_height = crop_height
        self.overlap = overlap
        self.g_slice_indices = self.get_slice_indices()
        self.g_slice_indices_y = self.get_slice_indices(626)
        self.seperate_y = len(self.g_slice_indices_y)
        self.inference_model = inference_model

    def get_offset(self, idx):
        cur_rank_y = idx // self.seperate
        if idx >= self.seperate * cur_rank_y:
            idx = idx - self.seperate * cur_rank_y
        idx_y = cur_rank_y

        return self.g_slice_indices[idx][0], self.g_slice_indices_y[idx_y][0]

    def get_slice_indices(self, full_size=1622):
        crop_s = self.crop_size
        over = self.overlap
        num_paths = math.ceil(full_size / crop_s)

        if full_size == 626:
            if self.crop_height == 0:
                return [[0, 626]]
            crop_s = self.crop_height
            over = 30
        else:
            self.seperate = num_paths

        slices = []
        for i in range(num_paths):
            start = max(crop_s * i - over, 0)
            end = start + crop_s
            if end > full_size:
                end = full_size
                start = end - crop_s

            slices.append([start, end])

        return slices

    def get_input_img(self, sample, crop=False):
        sample = tf.io.parse_single_example(sample, data_processing.image_feature_description)

        image = tf.image.decode_png(sample["image"])

        train_imgs = []
        small_imgs = []
        if crop:
            for start_y, end_y in self.g_slice_indices_y:
                for start_x, end_x in self.g_slice_indices:
                    small_img = image[start_y:end_y, start_x: end_x, :]
                    croped, _, ratio = utils.resize_and_pad_image(small_img, jitter=None)
                    train_imgs.append(tf.expand_dims(croped, axis=0))
                    small_imgs.append(small_img)

            return [tf.keras.applications.resnet.preprocess_input(i) for i in train_imgs], image, ratio

        else:
            train_img, _, ratio = utils.resize_and_pad_image(image, jitter=None)
            train_img = tf.keras.applications.resnet.preprocess_input(train_img)
            return tf.expand_dims(train_img, axis=0), image, ratio

    def revert_bboxes(self, boxes, idx):
        offset_x, offset_y = self.get_offset(idx)
        return tf.stack([
            boxes[idx, :, 0] + offset_x,
            boxes[idx, :, 1] + offset_y,
            boxes[idx, :, 2] + offset_x,
            boxes[idx, :, 3] + offset_y,
        ], axis=-1)

    def detect_single_image(self, sample, show=False):
        all_boxes = []
        all_scores = []
        all_classes = []

        input_img, image, ratio = self.get_input_img(sample, crop=True)

        if show:
            for c in range(self.seperate_y):
                start = self.seperate * c
                end = start + self.seperate
                img_up = tf.concat(input_img[start:end], 2)
                img_up = tf.image.resize(img_up, (626, 1622))
                cv2_imshow(img_up.numpy()[0] + 123)

        detections = self.inference_model.predict_on_batch(tf.concat(input_img, 0))

        boxes = detections.nmsed_boxes / ratio
        for i, valids in enumerate(detections.valid_detections):
            if valids > 0:
                for j in range(valids):
                    all_boxes.append(self.revert_bboxes(boxes, i)[j])

                all_classes.append(detections.nmsed_classes[i][:valids])
                all_scores.append(detections.nmsed_scores[i][:valids])

        len_detections = len(all_boxes)

        input_img, image, ratio = self.get_input_img(sample, crop=False)
        detections = self.inference_model.predict(input_img)
        num_detections = detections.valid_detections[0]
        big_size_boxes = []
        big_size_classes = []
        big_size_scores = []
        if num_detections:
            big_size_boxes.append(detections.nmsed_boxes[0][:num_detections] / ratio)
            big_size_scores.append(detections.nmsed_scores[0][:num_detections])
            big_size_classes.append(detections.nmsed_classes[0][:num_detections])

        if len_detections:
            all_boxes = tf.stack(all_boxes)
            all_scores = tf.concat(all_scores, 0)
            all_classes = tf.concat(all_classes, 0)
            if num_detections:
                all_boxes = tf.concat([all_boxes, tf.concat(big_size_boxes, 0)], 0)
                all_classes = tf.concat([all_classes, tf.concat(big_size_classes, 0)], 0)
                all_scores = tf.concat([all_scores, tf.concat(big_size_scores, 0)], 0)

            preds = tf.image.non_max_suppression(
                all_boxes,
                all_scores,
                100,
                iou_threshold=0.3,
                score_threshold=0.5,
            )
            preds = preds.numpy()

            if len(preds):
                return (image,
                        tf.gather(all_boxes, preds),
                        tf.gather(all_scores, preds),
                        tf.gather(all_classes, preds))

        return image, all_boxes, all_scores, all_classes


def get_inference_model():
    num_of_classes = 7
    model = m.RetinaNet(num_of_classes, backbone="densenet121")
    model.compile(optimizer="adam", loss=losses.RetinaNetLoss(num_of_classes))

    # Trick: fit model first so the model can load the weight
    model.fit(np.random.rand(1, 896, 2304,3), np.random.rand(1, 386694, 5))
    image = tf.keras.Input(shape=[None, None, 3], name="image")
    model.load_weights("./weight_dense.h5")
    predictions = model(image, training=False)
    detections = m.DecodePredictions(confidence_threshold=0.5,
                                     num_classes=num_of_classes,
                                     max_detections_per_class=10,
                                     nms_iou_threshold=0.5,
                                     verbose=0)(image, predictions)

    inference_model = tf.keras.Model(inputs=image, outputs=detections)

    return inference_model


def get_test_data_info(input_path):
    id_list = os.listdir(input_path)
    id_list = sorted([int(c.split(".")[0]) for c in id_list])

    data_info = [
        {
            "bbox": [[0, 0, 0, 0]],
            "label": [0],
            "id": x,
        } for x in id_list
    ]

    return data_info


if __name__ == "__main__":
    # Make prediction
    input_path = "/data/images"
    input_path = "/home/ubuntu/Documents/za_traffic_2020/traffic_public_test/images"

    TFRECORDS_FILE_PRIVATE_TEST = "./images_private_test.tfrecords"

    # Get list of test images
    data_info = get_test_data_info(input_path)

    if not os.path.isfile(TFRECORDS_FILE_PRIVATE_TEST):
        print("- tfrecords file not found, create new one")
        data_processing.write_tfrecords(data_info, TFRECORDS_FILE_PRIVATE_TEST, input_path)

    test_dataset = tf.data.TFRecordDataset(TFRECORDS_FILE_PRIVATE_TEST)

    # Create submission.json
    submission = []
    idx = 0
    predictor = Prediction(get_inference_model())

    for sample in test_dataset:
        image, boxes, scores, classes = predictor.detect_single_image(sample)
        if not isinstance(boxes, list):
            boxes = boxes.numpy()
            scores = scores.numpy()
            classes = classes.numpy()

        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box
            xywh = [x1, y1, x2 - x1, y2 - y1]
            score = scores[i]
            cls = classes[i]
            submission.append({
                "image_id": data_info[idx]["id"],
                "category_id": int(cls),
                "bbox": [float(z) for z in xywh],
                "score": float(score),
            })

        utils._print_progress("{}/{}".format(idx, 585))
        idx += 1

    with open("submission.json", "w") as f:
        json.dump(submission, f, indent=2)
