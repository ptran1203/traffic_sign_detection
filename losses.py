import tensorflow as tf

class RetinaNetBoxLoss(tf.losses.Loss):
    """Implements Smooth L1 loss"""

    def __init__(self, delta):
        super(RetinaNetBoxLoss, self).__init__(
            reduction="none", name="RetinaNetBoxLoss"
        )
        self._delta = delta

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference ** 2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)

class RetinaNetClassificationLoss(tf.losses.Loss):
    """Implements Focal loss"""

    def __init__(self, alpha, gamma, label_smoothing):
        super(RetinaNetClassificationLoss, self).__init__(
            reduction="none", name="RetinaNetClassificationLoss"
        )
        self._alpha = alpha
        self._gamma = gamma
        self._label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)

        if self._label_smoothing:
            alpha = tf.where(tf.greater(y_true, 0.91), self._alpha, (1.0 - self._alpha))
            pt = tf.where(tf.greater(y_true, 0.91), probs, 1 - probs)
        else:
            alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
            pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)

        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

class RetinaNetLoss(tf.losses.Loss):
    """Wrapper to combine both the losses"""

    def __init__(self, num_classes=80, alpha=0.25,
                gamma=2.0, delta=1.0, label_smoothing=True,
                ):
        super(RetinaNetLoss, self).__init__(reduction="auto", name="RetinaNetLoss")
        self._clf_loss = RetinaNetClassificationLoss(alpha, gamma, label_smoothing)
        self._box_loss = RetinaNetBoxLoss(delta)
        self._num_classes = num_classes
        self._label_smoothing = label_smoothing
        self._factor = 0.1
        self._max_label = (1 - self._factor)

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]
        cls_labels = tf.one_hot(
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self._num_classes,
            dtype=tf.float32,
        )

        if self._label_smoothing:
            cls_labels = _smooth_labels(cls_labels)

        cls_predictions = y_pred[:, :, 4:]
        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)
        clf_loss = self._clf_loss(cls_labels, cls_predictions)
        box_loss = self._box_loss(box_labels, box_predictions)

        if self._label_smoothing:
            clf_loss = tf.where(tf.greater(ignore_mask, 0.8), 0.0, clf_loss)
            box_loss = tf.where(tf.equal(positive_mask, 1), box_loss, 0.0)
        else:
            clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
            box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)

        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)

        loss = clf_loss + box_loss

        return loss


def _smooth_labels(labels):
    """Apply label smoothing"""
    factor = 0.1
    labels = labels * (1 - factor)
    labels = labels + (factor / tf.cast(tf.shape(labels)[1], tf.float32))

    return labels
