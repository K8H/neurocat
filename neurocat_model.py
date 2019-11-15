import os
import uuid

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from scipy.misc import toimage


class NeuroCatModel(object):
    INPUT_TENSOR_NAME = 'input:0'
    OUTPUT_TENSOR_NAME = 'output:0'
    LABELS_TENSOR_NAME = 'label:0'
    LOSS_TENSOR_NAME = 'loss:0'
    LABELS = dict(
        zip(range(10), ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']))

    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, directory, graph_filename):
        self._load_graph(directory, graph_filename)

    def _load_graph(self, directory, graph_filename):
        """Loads pretrained model from a graph checkpoints.

        Params
        ------
        directory: string
            Path to a folder where checkpoints will be saved. It creates directory if it doesn't exist.
        filename: string
            File name under which checkpoint are saved.

        TODO set exponential moving average for testing purpose
        """
        print('Loading model...')
        self.sess = tf.Session()
        self.graph = tf.get_default_graph()

        with self.graph.as_default():
            # TODO set EMA
            var_avgs = tf.train.ExponentialMovingAverage(0.999)
            ema_vars_to_restore = var_avgs.variables_to_restore()

            with self.sess.as_default():
                saver = tf.train.import_meta_graph(os.path.join(directory, graph_filename + '.ckpt.meta'))
                # saver = tf.train.Saver(ema_vars_to_restore)

                saver.restore(self.sess, tf.train.latest_checkpoint(directory))
                self.sess.run(tf.global_variables_initializer())

        self.graph.finalize()

        print('Model loading complete!')

    def predict(self, x_test):
        """Calculates predictions on a trained model on a given input images.

        Pretrained model has to consist of 'input:0' and 'output:0' nodes which are inferred from a graph.

        Params
        ------
        x_test: List<numpy.ndarray>
            List of test images.

        Returns
        -------
        List<numpy.ndarray>
            List of probability values for every class.
        """
        input_tensor = self.graph.get_tensor_by_name("input:0")
        output_tensor = self.graph.get_tensor_by_name("output:0")
        output = self.sess.run(output_tensor, feed_dict={input_tensor: x_test})

        return output

    def label_names_mapper(self, labels):
        """A mapper from prediction values to human readable names.

        Class names: airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'

        Params
        ------
        labels: List<int>
            List of target classes.

        Returns
        -------
        List<string>
            List of human readable names.

        Raises
        ------
        TypeError
            If label values are not of int type.
        """
        if type(labels) != list and type(labels) != np.ndarray:
            if type(labels) != int:
                raise TypeError('Label input type must be int')
            labels = [labels]
        elif not all(isinstance(i, int) for i in labels):
            raise TypeError('All label values types must be int')

        label_names = [self.LABELS.get(label) for label in labels]
        return label_names

    def plot(self, input, output, plot_name=None):
        """Create a figure of an input image with output label name.

        Images are saved in a directory 'figures'.

        Params
        ------
        input: numpy.ndarray
            3D array of input image
        output: int
            Target value
        plot_name: string
            Name of the plot - optional. If not given a figure is assigned with unique id (uuid).

        Returns
        -------
        string
            Name of the file under which an image was saved.
        """
        directory = 'figures'
        if not os.path.exists(directory):
            os.makedirs(directory)

        if plot_name is None:
            plot_name = str(uuid.uuid1())
        filename = os.path.join(directory, plot_name)

        plt.figure(1)
        plt.imshow(toimage(input))
        plt.title(self.label_names_mapper(output))
        plt.savefig(filename)

        return filename
