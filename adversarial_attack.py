import tensorflow as tf

EPS = 0.25  # optimal value for cifar10 task


def fast_gradient_sign_method(model, img, target):
    """Creates perturbed image with a fast gradient sign method.

    From model acquires needed tensors, namely input, loss, labels. Based on an input it calculates gradients from
    a loss function.

    Parameters
    ----------
     model: Model
        Pretrained model of any differential kind, which have 'input:0', 'loss:0' and 'labels:0' graph nodes.
    img: numpy.ndarray
        Image that needs to be perturbated.
    target: int
        Target label.

    Returns
    -------
    numpy.ndarray
        Perturbated image.

    Raises
    ------
    KeyError
        If any of graph nodes does not correspond to tensors in this graph.
    """
    img = img.reshape(1, 32, 32, 3)
    target_onehot = tf.keras.utils.to_categorical(target, num_classes=10)

    try:
        input_tensor = model.graph.get_tensor_by_name(model.INPUT_TENSOR_NAME)
        loss_tensor = model.graph.get_tensor_by_name(model.LOSS_TENSOR_NAME)
        labels_tensor = model.graph.get_tensor_by_name(model.LABELS_TENSOR_NAME)
    except  KeyError:


    grad = model.sess.run(
        tf.gradients(loss_tensor, input), feed_dict={input_tensor: img, labels_tensor: target_onehot})

    perturbation = EPS * tf.sign(grad)

    adv_img = img + perturbation

    return adv_img
