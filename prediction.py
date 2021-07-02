import math
import tensorflow as tf
import utils
import os
import model as m
import losses
import numpy as np
import json
import argparse
import datetime
import glob
import cv2
from tqdm import tqdm
from utils import visualize_detections

LABEL_MAP = {
    1: "No entry",
    2: "No parking / waiting",
    3: "No turning",
    4: "Max Speed",
    5: "Other prohibition signs",
    6: "Warning",
    7: "Mandatory",
}

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

    def get_input_img(self, image, crop=False, crop_size=512):
        image = tf.convert_to_tensor(image)

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

    def detect_single_image(self, image, crop_sizes=[], tiling=False):
        all_boxes = []
        all_scores = []
        all_classes = []

        sboxes, sscores, sclasses = [], [], []

        if not crop_sizes:
            crop_sizes = [1024]

        detected = False
        if tiling:
            input_img, image, ratio = self.get_input_img(image, crop=True, crop_size=self.tiling_size)

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

        small_detections = len(sboxes)

        for crop_size in crop_sizes:
            input_img, image, ratio = self.get_input_img(image, crop=False, crop_size=crop_size)
            detections = self.inference_model.predict(input_img)
            num_detections = detections.valid_detections[0]

            if num_detections:
                detected = True
                scores = detections.nmsed_scores[0][:num_detections]

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


def get_inference_model(weight_path, backbone="resnet50"):
    num_of_classes = 7
    model = m.RetinaNet(num_of_classes, backbone=backbone)
    model.compile(optimizer="adam", loss=losses.RetinaNetLoss(num_of_classes))
    model.build((1, None, None, 3))
    image = tf.keras.Input(shape=[None, None, 3], name="image")
    model.load_weights(weight_path)
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


def run_prediction(args):
    input_path, output_path, weight, save_dir = (
        args.input, args.output, args.weight, args.save_dir)

    backbone = weight.split("_")[-1].replace(".h5", "")
    crop_sizes = list(map(int, args.scales.split(",")))
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("/".join(output_path.split("/")[:-1]), exist_ok=True)

    if output_path.split(".")[-1] != "json":
        raise ValueError("Output file should be json format")

    # Get list of test images
    if os.path.isdir(input_path):
        image_files = glob.glob(os.path.join(input_path, '*'))
    else:
        # it's file
        image_files = [input_path]

    print(f"Test on {len(image_files)} images")

    # Create submission.json
    submission = []
    predictor = Prediction(get_inference_model(weight, backbone))

    start = datetime.datetime.now()
    for file_path in tqdm(image_files):
        image, boxes, scores, classes = predictor.detect_single_image(
            cv2.imread(file_path)[..., ::-1],
            crop_sizes=crop_sizes,
            tiling=args.tiling
        )
        if not isinstance(boxes, list):
            boxes = boxes.numpy()
            scores = scores.numpy()
            classes = classes.numpy()

        if save_dir:
            save_path = os.path.join(save_dir, file_path.split("/")[-1])
            cls_name = [
                LABEL_MAP[int(x)] for x in classes
            ]
            visualize_detections(image, boxes, cls_name, scores, save_path=save_path)

        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box
            xywh = [x1, y1, x2 - x1, y2 - y1]
            score = scores[i]
            cls = classes[i]
            submission.append({
                "image_id": file_path,
                "category_id": int(cls),
                "bbox": [float(z) for z in xywh],
                "score": float(score),
            })

    print("Predict in {}".format(datetime.datetime.now() - start))

    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)

    print("Submission saved at {}".format(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Traffic sign detection')
    parser.add_argument("--input",
                        metavar="I", type=str, default="/data/images",
                        help="Path to input images")
    parser.add_argument("--output", metavar="O", type=str,
                        default="/data/result/submission.json", help="Output file path")
    parser.add_argument("--weight", metavar="W", type=str,
                        default="pretrained_densenet121", help="Weight path")
    parser.add_argument("--save-dir", type=str, default="/content/infernece_images")
    parser.add_argument("--tiling", action="store_true")
    parser.add_argument("--scales", type=str, default="1024", help="Separated by comma ','")

    args = parser.parse_args()

    print(args)

    run_prediction(args)
