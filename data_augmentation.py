import tensorflow as tf
import numpy as np

size = 5
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size
kernel_motion_blur = np.expand_dims(kernel_motion_blur, axis=-1)
kernel_motion_blur = np.repeat(kernel_motion_blur, repeats=3, axis=-1)
kernel_motion_blur = np.expand_dims(kernel_motion_blur, axis=-1)
kernel_motion_blur = tf.cast(kernel_motion_blur, tf.float32)


def random_flip_horizontal(image, boxes):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes


def random_adjust_contrast(image):
    if tf.random.uniform(()) > 0.5:
        factor = tf.random.uniform((), 0.5, 2.0)
        return tf.image.adjust_contrast(image, factor)

    return image


def random_adjust_brightness(image):
    if tf.random.uniform(()) > 0.5:
        return tf.image.random_brightness(image, 0.3)

    return image


def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


def random_gaussian_blur(img):
    if tf.random.uniform(()) > 0.5:
        img = tf.cast(img, dtype=tf.float32)
        if tf.random.uniform(()) > 0.5:
            kernel = _gaussian_kernel(7, 3, 3, img.dtype)
        else:
            kernel = kernel_motion_blur
        img = tf.nn.depthwise_conv2d(img[None], kernel, [1, 1, 1, 1], "SAME")

        return tf.cast(img[0], dtype=tf.uint8)

    return img