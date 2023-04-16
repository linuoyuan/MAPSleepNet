import tensorflow as tf

def softmax_cross_entrophy_loss(logits, targets):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits,
        targets,
        name="cross_entropy_per_example"
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
    return cross_entropy_mean


def softmax_seq_loss_by_example(logits, targets, batch_size, seq_length):
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [targets],
        [tf.ones([batch_size * seq_length])])
    seq_cross_entropy_mean = tf.reduce_sum(loss) / batch_size
    return seq_cross_entropy_mean


def focal_loss(labels, logits, gamma=2.0, alpha=4.0):
    """
    focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: logits is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)

    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal Loss for Dense Object Detection, 130(4), 485–491.
    https://doi.org/10.1016/j.ajodo.2005.02.022

    :param labels: ground truth labels, shape of [batch_size]
    :param logits: model's output, shape of [batch_size, num_cls]
    :param gamma:
    :param alpha:
    :return: shape of [batch_size]
    """
    epsilon = 1.e-9
#    labels = tf.to_int64(labels)
    labels = tf.convert_to_tensor(labels)
    logits = tf.convert_to_tensor(logits, tf.float32)
    num_cls = logits.shape[1]

    model_out = tf.add(logits, epsilon)
    onehot_labels = tf.one_hot(labels, num_cls)
    ce = tf.multiply(onehot_labels, -tf.math.log(model_out))
    weight = tf.multiply(onehot_labels, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    # reduced_fl = tf.reduce_sum(fl, axis=1)  # same as reduce_max
    return reduced_fl