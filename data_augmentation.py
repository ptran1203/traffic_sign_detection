import tensorflow as tf
import numpy as np
import math

size = 3
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size
kernel_motion_blur = np.expand_dims(kernel_motion_blur, axis=-1)
kernel_motion_blur = np.repeat(kernel_motion_blur, repeats=3, axis=-1)
kernel_motion_blur = np.expand_dims(kernel_motion_blur, axis=-1)
kernel_motion_blur = tf.cast(kernel_motion_blur, tf.float32)


def random_flip_horizontal(image, boxes, prob=0.5):
    if tf.random.uniform(()) > prob:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes


def random_adjust_contrast(image, prob=0.5):
    if tf.random.uniform(()) > prob:
        factor = tf.random.uniform((), 0.5, 2.0)
        return tf.image.adjust_contrast(image, factor)

    return image


def random_adjust_brightness(image, prob=0.5):
    if tf.random.uniform(()) > prob:
        return tf.image.random_brightness(image, 0.06)

    return image


def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


def random_gaussian_blur(img, prob=0.9):
    if tf.random.uniform(()) > prob:
        img = tf.cast(img, dtype=tf.float32)
        if tf.random.uniform(()) > 0.5:
            kernel = _gaussian_kernel(7, 3, 3, img.dtype)
        else:
            kernel = kernel_motion_blur
        img = tf.nn.depthwise_conv2d(img[None], kernel, [1, 1, 1, 1], "SAME")

        return tf.cast(img[0], dtype=tf.uint8)

    return img


# class Augmentor:
#     def __init__(
#         self,
#         blur=0.8,
#         horizontal_flip=0.5,
#         brightness=0.5,
#         contrast=0.5
#     ):

#     self.blur = blur
#     self.horizontal_flip = horizontal_flip
#     self.brightness = brightness
#     self.contrast = contrast

#     def augment(self, image, boxes, labels):
#         image, boxes = random_flip_horizontal(image, boxes, self.horizontal_flip)
#         image = random_adjust_brightness(image, self.brightness)
#         image = random_adjust_contrast(image, self.contrast)
#         image = random_gaussian_blur(image, self.blur)

#         return image, boxes, labels
