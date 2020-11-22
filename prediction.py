import math
import tensorflow as tf
import utils
try:
    from google.colab.patches import cv2_imshow
except ImportError:
    import cv2.imshow as cv2_imshow

class Prediction:
    def __init__(self, inference_model, crop_size=200, overlap=75):
        self.crop_size = crop_size
        self.overlap = overlap
        self.g_slice_indices = self.get_slice_indices()
        self.seperate = 0
        self.inference_model = inference_model

    def get_offset(self, idx):
        return self.g_slice_indices[idx][0], 0

    def get_slice_indices(self.img_width=1622):        
        crop_s = self.crop_size
        over = self.overlap
        num_paths = math.ceil(img_width/crop_s)

        if img_width == 626:
            crop_s = crop_height
            over=10
        else:
            self.seperate = num_paths

        slices = []
        for i in range(num_paths):
            start = max(crop_s * i - over, 0)
            end = start + crop_s
            if end > img_width:
                end = img_width
                start = end - crop_s

            slices.append([start, end])

        return slices

    def get_input_img(self, sample, crop=False):
        sample = tf.io.parse_single_example(sample,
            data_processing.image_feature_description)

        image = tf.image.decode_png(sample["image"])

        train_imgs = []
        small_imgs = []
        if crop:
            # for start_y, end_y in get_slice_indices(626):
            for start_x, end_x in self.g_slice_indices:
                small_img = image[:, start_x: end_x, :]
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
            boxes[idx,:, 0] + offset_x,
            boxes[idx,:, 1] + offset_y,
            boxes[idx,:, 2] + offset_x,
            boxes[idx,:, 3] + offset_y,
        ], axis= -1)

    def detect_single_image(self, sample, show=False):
        all_boxes = []
        all_scores = []
        all_classes = []

        input_img, image, ratio = self.get_input_img(sample, crop=True)

        if show:
            img_up = tf.concat(input_img,2)
            img_up = tf.image.resize(img_up, (626, 1622))
            cv2_imshow(img_up.numpy()[0] + 123)

        detections = self.inference_model.predict_on_batch(tf.concat(input_img, 0))

        boxes = detections.nmsed_boxes / ratio
        for i, valids in enumerate(detections.valid_detections):
            if valids > 0:
                for j in range(valids):
                    all_boxes.append(revert_bboxes(boxes, i)[j])

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
