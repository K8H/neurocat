import numpy as np


def one_class_prediction(predictions_onehot):
    """Converts onehot label format into label class value.

    Parameters
    ----------
     predictions_onehot: List<float32>
        Prediction values in onehot format

    Return
    ------
    List<float32>
        Class label values.
    """
    return np.argmax(predictions_onehot, axis=1).reshape((-1, 1))


def prediction_accuracy(predictions_onehot, labels):
    """Calculates prediction's accuracy.

    Converts onehot format into label class value and summarize all accurate predictions and normalize with number of
    all predictions.

    Parameters
    ----------
     predictions_onehot: List<float32>
        Prediction values in onehot format
    labels: List<float32>
        target labels

    Return
    ------
    float32
        Prediction accuracy
    """
    predictions = one_class_prediction(predictions_onehot)
    return np.sum(predictions == labels) / len(predictions)

