import tensorflow as tf

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
        return tf.image.random_brightness(
            image, 0.3
        )

    return image