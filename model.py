import os
import tensorflow as tf


class Model(object):

    def __init__(self, input_size, num_classes):
        """Initialize a Model instance.

        Prepare all placeholders, model structure, loss function, optimizer.

        Params
        ------
        input_size: [width, height, channels]
            Size of an input image
        num_classes: int
            Number of classification targets.
        """
        self.input_size = input_size
        self.num_classes = num_classes

        self.config = tf.ConfigProto(device_count={'GPU': 0})
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.input = tf.placeholder(tf.float32, [None] + self.input_size, name='input')
        self.label = tf.placeholder(tf.float32, [None, self.num_classes], name='label')
        self.output = self.model()
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.label), name='loss')
        self.optimization = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def model(self):
        """Compose a model structure.

        Convolutional input layer, 32 feature maps with a size of 3×3, a rectifier activation function
        Convolutional input layer, 32 feature maps with a size of 3×3, a rectifier activation function
        Max Pool layer with size 2×2
        Dropout set to 20%
        Convolutional input layer, 64 feature maps with a size of 3×3, a rectifier activation function
        Convolutional input layer, 64 feature maps with a size of 3×3, a rectifier activation function
        Max Pool layer with size 2×2
        Dropout set to 30%
        Convolutional input layer, 128 feature maps with a size of 3×3, a rectifier activation function
        Convolutional input layer, 128 feature maps with a size of 3×3, a rectifier activation function
        Max Pool layer with size 2×2
        Dropout set to 40%
        Flatten layer
        Fully connected layer with 128 units and a rectifier activation function
        Dropout set to 50%
        Fully connected output layer with 10 units and a softmax activation function

        Return
        ------
        tf.Tensor
            output tensor
        """
        conv1 = tf.layers.conv2d(inputs=self.input, filters=32, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.relu,
                                 name='conv1')
        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.relu,
                                 name='conv2')

        pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=[2, 2], name='pool1')
        pool1_dropout = tf.layers.dropout(inputs=pool1, rate=0.2, training=True, name='pool1_dropout')

        conv3 = tf.layers.conv2d(inputs=pool1_dropout, filters=64, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.relu, name='conv3')
        conv4 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.relu,
                                 name='conv4')

        pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=[2, 2], name='pool2')
        pool2_dropout = tf.layers.dropout(inputs=pool2, rate=0.3, training=True, name='pool2_dropout')

        conv5 = tf.layers.conv2d(inputs=pool2_dropout, filters=128, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.relu, name='conv5')
        conv6 = tf.layers.conv2d(inputs=conv5, filters=128, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.relu, name='conv6')

        pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=[2, 2], name='pool3')
        pool3_dropout = tf.layers.dropout(inputs=pool3, rate=0.4, training=True, name='pool3_dropout')

        flat = tf.layers.flatten(inputs=pool3_dropout, name='flat')
        fc1 = tf.layers.dense(inputs=flat, units=128, activation=tf.nn.relu, name='fc1')
        fc1_dropout = tf.layers.dropout(inputs=fc1, rate=0.5, training=True, name='fc1_dropout')
        fc2 = tf.layers.dense(inputs=fc1_dropout, units=self.num_classes, activation=None, name='fc2')

        output = tf.identity(fc2, name='output')

        return output

    def train(self, x_train, y_train, learning_rate):
        """Fits the model to data.

        One iteration of a training, computes loss value on given values and optimizes with an initialized optimizer.

        Params
        ------
        x_train: List<numpy.ndarray>
            List of train images.
        y_train: List<numpy.ndarray>
            List of onehot format of target classes.
        learning_rate: float32
            Adjustment of weights with respect to the loss gradient.

        Returns
        -------
        float32
            Loss value of training iteration.
        """
        _, train_loss = self.sess.run(
            [self.optimization, self.loss],
            feed_dict={self.input: x_train, self.label: y_train, self.learning_rate: learning_rate})

        return train_loss

    def predict(self, x_test):
        """Calculates predictions on a trained model on a given input images.

        Params
        ------
        x_test: List<numpy.ndarray>
            List of test images.

        Returns
        -------
        List<numpy.ndarray>
            List of probability values for every class.
        """
        return self.sess.run(self.output, feed_dict={self.input: x_test})

    def save_graph_checkpoints(self, directory, filename):
        """Save graph checkpoints of a trained model to a given directory/filename

        Params
        ------
        directory: string
            Path to a folder where checkpoints will be saved. It creates directory if it doesn't exist.
        filename: string
            File name under which checkpoint are saved.
        """
        print('Saving model %s...' % filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        ckpt_filepath = os.path.join(directory, filename + '.ckpt')

        var_averages = tf.train.ExponentialMovingAverage(0.0)
        var_to_restore = var_averages.variables_to_restore()
        saver = tf.train.Saver(var_to_restore)

        saver.save(self.sess, ckpt_filepath)

        print('Model %s saved successfully!' % filename)
