import math
import tensorflow as tf
import utils
import data_processing
import os
import model as m
import losses
import numpy as np
import json
import argparse
import datetime

try:
    from google.colab.patches import cv2_imshow
except ImportError:
    try:
        import cv2.imshow as cv2_imshow
    except ImportError:
        def cv2_imshow(img):
            return

class Prediction:
    def __init__(self,
    inference_model,
    crop_size=200,
    image_height=626,
    image_width=1622,
    crop_height=300,
    overlap=75,
    dynamic_size=False,
    tiling_size=968):
        self.crop_size = crop_size
        self.crop_height = crop_height
        self.image_width = image_width
        self.tiling_size = tiling_size
        self.image_height = image_height
        self.overlap = overlap
        self.dynamic_size = dynamic_size
        self.g_slice_indices = self.get_slice_indices(image_width)
        self.g_slice_indices_y = self.get_slice_indices(image_height)
        self.seperate_y = len(self.g_slice_indices_y)
        self.inference_model = inference_model

    def set_height(self, height):
        self.image_height = height
        self.crop_height = height // 4
        self.g_slice_indices_y = self.get_slice_indices(height)

    def set_width(self, width):
        self.image_width = width
        self.crop_size = width // 4
        self.g_slice_indices = self.get_slice_indices(width)

    def get_offset(self, idx):
        cur_rank_y = idx // self.seperate
        if idx >= self.seperate * cur_rank_y:
            idx = idx - self.seperate * cur_rank_y
        idx_y = cur_rank_y

        return self.g_slice_indices[idx][0], self.g_slice_indices_y[idx_y][0]

    def get_slice_indices(self, full_size):
        crop_s = self.crop_size
        over = self.overlap
        num_paths = math.ceil(full_size / crop_s)

        if full_size == self.image_height and full_size != self.image_width:
            if self.crop_height == 0:
                return [[0, self.image_height]]
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

    def get_input_img(self, sample, crop=False, crop_size=512):
        sample = tf.io.parse_single_example(sample, data_processing.image_feature_description)

        image = tf.image.decode_png(sample["image"])

        if self.dynamic_size:
            shape = image.shape
            self.set_height(shape[0])
            self.set_width(shape[1])

        train_imgs = []
        small_imgs = []

        if crop:
            ratio = 0
            for start_y, end_y in self.g_slice_indices_y:
                for start_x, end_x in self.g_slice_indices:
                    small_img = image[start_y:end_y, start_x: end_x, :]
                    if start_x + self.crop_size > self.image_width:
                        start_x = self.image_width -  self.crop_size
                    if start_y + self.crop_height > self.image_height:
                        start_y = self.image_height - self.crop_height
                    
                    small_img = tf.slice(image, [start_y, start_x, 0], [self.crop_height, self.crop_size, 3])

                    croped, _, ratio = utils.resize_and_pad_image(small_img,
                                                                  crop_size,
                                                                  crop_size, jitter=None)
                    train_imgs.append(tf.expand_dims(croped, axis=0))
                    small_imgs.append(small_img)

            return [tf.keras.applications.resnet.preprocess_input(i) for i in train_imgs], image, ratio

        else:
            train_img, _, ratio = utils.resize_and_pad_image(image,
                                                             crop_size,
                                                             crop_size,
                                                             jitter=None)
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

    @staticmethod
    def big_box_filter(image, boxes, scores, classes, threshold=.12):
        img_h, img_w, _ = image.shape
        fboxes, fscores, fclasses = [], [], []
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w / img_w <= threshold and h / img_h <= threshold:
                fboxes.append(box)
                fscores.append(score)
                fclasses.append(cls)

        return tf.stack(fboxes), tf.stack(fscores), tf.stack(fclasses)


    def detect_single_image(self, sample, crop_sizes=[], show=False, tiling=False):
        all_boxes = []
        all_scores = []
        all_classes = []

        sboxes, sscores, sclasses = [], [], []

        if not crop_sizes:
            crop_sizes = [1024, 1280, 1420]

        detected = False
        if tiling:
            input_img, image, ratio = self.get_input_img(sample, crop=True, crop_size=self.tiling_size)

            detections = self.inference_model.predict_on_batch(tf.concat(input_img, 0))

            boxes = detections.nmsed_boxes / ratio
            for i, valids in enumerate(detections.valid_detections):
                if valids > 0:
                    for j in range(valids):
                        sboxes.append(self.revert_bboxes(boxes, i)[j])

                    sclasses.append(detections.nmsed_classes[i][:valids])
                    sscores.append(detections.nmsed_scores[i][:valids])

            if len(sboxes):
                sboxes = tf.stack(sboxes)
                sscores = tf.concat(sscores, 0)
                sclasses = tf.concat(sclasses, 0)

                sboxes, sscores, sclasses = self.big_box_filter(image,
                                                        sboxes, sscores, sclasses)

        small_detections = len(sboxes)
        show and print(f"Found {small_detections} objects in small parts - {sscores}")

        for crop_size in crop_sizes:
            input_img, image, ratio = self.get_input_img(sample, crop=False, crop_size=crop_size)
            detections = self.inference_model.predict(input_img)
            num_detections = detections.valid_detections[0]

            if num_detections:
                detected = True
                scores = detections.nmsed_scores[0][:num_detections]
                show and print(f"Found {num_detections} objects at scale {crop_size} - {scores}")

                all_boxes.append(detections.nmsed_boxes[0][:num_detections] / ratio)
                all_scores.append(scores)
                all_classes.append(detections.nmsed_classes[0][:num_detections])

        if small_detections:       
            if len(all_classes):
                all_boxes = tf.concat(all_boxes, 0)
                all_scores = tf.concat(all_scores, 0)
                all_classes = tf.concat(all_classes, 0)

                if detected:
                    all_boxes = tf.concat([all_boxes, sboxes ], 0)
                    all_scores = tf.concat([all_scores, sscores], 0)
                    all_classes = tf.concat([all_classes, sclasses], 0)
            else:
                all_boxes = sboxes
                all_scores =  sscores
                all_classes = sclasses


        elif detected:
            all_boxes = tf.concat(all_boxes, 0)
            all_scores = tf.concat(all_scores, 0)
            all_classes = tf.concat(all_classes, 0)

        if detected or small_detections:
            selected_indices = tf.image.non_max_suppression(
                all_boxes,
                all_scores,
                50,
                iou_threshold=0.1,
                score_threshold=0.5,
            )

            selected_indices = selected_indices.numpy()

            if len(selected_indices):
                return (image,
                        tf.gather(all_boxes, selected_indices),
                        tf.gather(all_scores, selected_indices),
                        tf.gather(all_classes, selected_indices))

        return image, all_boxes, all_scores, all_classes


def get_inference_model():
    num_of_classes = 7
    model = m.RetinaNet(num_of_classes, backbone="densenet121")
    model.compile(optimizer="adam", loss=losses.RetinaNetLoss(num_of_classes))

    # Trick: fit model first so the model can load the weight
    model.fit(np.random.rand(1, 896, 2304, 3), np.random.rand(1, 386694, 5))
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

def combine_prediction(
    prediction_1,
    prediction_2,
    weight_1=1,
    max_detections=50,
    iou_threshold=0.5,
    score_threshold=0.65):
    boxes_1, scores_1, classes_1 = prediction_1
    boxes_2, scores_2, classes_2 = prediction_2

    weight_2 = 1 - weight_1
    highest = max(weight_1, weight_2)
    score_threshold *= highest
    
    if not len(scores_1) and len(scores_2):
        scores = scores_2 * weight_2
        boxes = boxes_2
        classes = classes_2
    elif not len(scores_2) and len(scores_1):
        scores = scores_1 * weight_1
        boxes = boxes_1
        classes = classes_1

    elif not len(scores_1) and not len(scores_2):
        return boxes_1, scores_1, classes_1

    else:
        scores_1 *= weight_1
        scores_2 *= weight_2

        boxes = tf.concat([boxes_1, boxes_2], 0)
        scores = tf.concat([scores_1, scores_2], 0)
        classes = tf.concat([classes_1, classes_2], 0)

    selected_indices =  tf.image.non_max_suppression(
        boxes,
        scores,
        max_detections,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )

    return (tf.gather(boxes, selected_indices),
            tf.gather(scores / highest, selected_indices),
            tf.gather(classes, selected_indices))

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
    parser = argparse.ArgumentParser(description='Traffic sign detection')
    parser.add_argument("--input", dest="input_path",
                        metavar="I", type=str, default="/data/images",
                        help="Path to input images")
    parser.add_argument("--test_file", dest="test_file", default="./images_private_test.tfrecords",
                        metavar="F", type=str, help="Tfrecords test file",)
    parser.add_argument("--output", dest="output_path", metavar="O", type=str,
                        default="/data/result/submission.json", help="Output file path")
    args = parser.parse_args()

    # Make prediction
    input_path = args.input_path
    output_path = args.output_path

    if output_path.split(".")[-1] != "json":
        raise("Output file should be json format")

    TFRECORDS_FILE_PRIVATE_TEST = args.test_file

    # Get list of test images
    data_info = get_test_data_info(input_path)

    print("Test on {} images".format(len(data_info)))

    print("Create tfrecords dataset")
    data_processing.write_tfrecords(data_info, TFRECORDS_FILE_PRIVATE_TEST, input_path)

    test_dataset = tf.data.TFRecordDataset(TFRECORDS_FILE_PRIVATE_TEST)

    # Create submission.json
    submission = []
    idx = 0
    predictor = Prediction(get_inference_model())

    print("Start predict...")
    start = datetime.datetime.now()
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

    print("Predict in {}".format(datetime.datetime.now() - start))

    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)

    print("Submission saved at {}".format(output_path))
