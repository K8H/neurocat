import os
import unittest
import libs
import adversarial_attack

import dataset as dataset
import tensorflow as tf

from model import Model
from neurocat_model import NeuroCatModel


class NeuroCatModelTest(tf.test.TestCase):

    DIRECTORY = 'saved_models/cifar10_%s'   # Insert graph's ID
    GRAPH_FILENAME = 'frozen_inference_graph'
    EPOCHS = 1

    def _train_load_model(self, epochs=None):
        if epochs is None:
            epochs = self.EPOCHS
        graph_dir = self.DIRECTORY % str(epochs)

        input_size = dataset.input_size

        model = Model(input_size, dataset.num_classes)
        model.train(dataset.x_train, dataset.y_train_onehot, 0.001)
        model.save_graph_checkpoints(graph_dir, self.GRAPH_FILENAME)

        trained_model = NeuroCatModel(input_size)
        trained_model(directory=graph_dir, graph_filename=self.GRAPH_FILENAME)

        return model, trained_model

    def test_is_class_callable(self):
        self.assertTrue(callable(NeuroCatModel(3)))

    def test_load_graph_tensor_names(self):
        _, trained_model = self._train_load_model()

        with self.test_session(graph=trained_model.graph):
            trained_model.graph.get_tensor_by_name('input:0')
            trained_model.graph.get_tensor_by_name('conv1/Conv2D:0')
            trained_model.graph.get_tensor_by_name('conv2/Conv2D:0')
            trained_model.graph.get_tensor_by_name('pool1/MaxPool:0')
            trained_model.graph.get_tensor_by_name('pool1_dropout/dropout/Shape:0')
            trained_model.graph.get_tensor_by_name('conv3/Conv2D:0')
            trained_model.graph.get_tensor_by_name('conv4/Conv2D:0')
            trained_model.graph.get_tensor_by_name('pool2/MaxPool:0')
            trained_model.graph.get_tensor_by_name('pool2_dropout/dropout/Shape:0')
            trained_model.graph.get_tensor_by_name('conv5/Conv2D:0')
            trained_model.graph.get_tensor_by_name('conv6/Conv2D:0')
            trained_model.graph.get_tensor_by_name('pool3/MaxPool:0')
            trained_model.graph.get_tensor_by_name('pool3_dropout/dropout/Shape:0')
            trained_model.graph.get_tensor_by_name('flat/Reshape:0')
            trained_model.graph.get_tensor_by_name('fc1/MatMul:0')
            trained_model.graph.get_tensor_by_name('fc1_dropout/dropout/Shape:0')
            trained_model.graph.get_tensor_by_name('fc2/MatMul:0')
            trained_model.graph.get_tensor_by_name('output:0')
            trained_model.graph.get_tensor_by_name('output:0')

    def test_load_graph_wrong_directory(self):
        graph_dir = self.DIRECTORY % 'wrong'
        trained_model = NeuroCatModel(3)

        self.assertRaises(OSError, trained_model, directory=graph_dir, graph_filename=self.GRAPH_FILENAME)

    def test_load_graph_wrong_input_shape(self):
        _, trained_model = self._train_load_model()

        with self.test_session(graph=trained_model.graph) as sess:
            input_vector = trained_model.graph.get_tensor_by_name('input:0')
            output_vector = trained_model.graph.get_tensor_by_name("output:0")
            with self.assertRaises(ValueError):
                sess.run(output_vector, feed_dict={input_vector: dataset.x_test.reshape((-1, 1))})

    def test_load_graph_predict(self):
        # TODO set Exponential Moving Average in training and inference
        model, trained_model = self._train_load_model()

        x_test = dataset.x_test
        y_test = dataset.y_test

        model_predictions_onehot = model.predict(x_test)
        model_accuracy = libs.prediction_accuracy(model_predictions_onehot, y_test)

        trained_model_predictions_onehot = trained_model.predict(x_test)
        trained_model_accuracy = libs.prediction_accuracy(trained_model_predictions_onehot, y_test)

        self.assertEqual(model_accuracy, trained_model_accuracy)

    def test_label_names_mapper(self):
        _, trained_model = self._train_load_model()
        labelname = trained_model.label_names_mapper(5)

        self.assertEqual(labelname, ['dog'])

    def test_label_names_mapper_list(self):
        _, trained_model = self._train_load_model()
        labelname = trained_model.label_names_mapper([5, 7, 1])

        self.assertEqual(labelname, ['dog', 'horse', 'automobile'])

    def test_label_names_mapper_str(self):
        _, trained_model = self._train_load_model()
        self.assertRaises(TypeError, trained_model.label_names_mapper, 'bunny')

    def test_label_names_mapper_list_str(self):
        _, trained_model = self._train_load_model()
        self.assertRaises(TypeError, trained_model.label_names_mapper, ['bunny', 2, '2.5'])

    def test_plot(self):
        _, trained_model = self._train_load_model()

        x_test = dataset.x_test

        trained_model_predictions_onehot = trained_model.predict(x_test)
        predictions = libs.one_class_prediction(trained_model_predictions_onehot)

        filename = trained_model.plot(x_test[1], predictions.item(1))
        self.assertTrue(os.path.isfile(filename + '.png'))

    def test_plot_wrong_input(self):
        _, trained_model = self._train_load_model()

        x_test = dataset.x_test

        trained_model_predictions_onehot = trained_model.predict(x_test)
        predictions = libs.one_class_prediction(trained_model_predictions_onehot)

        filename = trained_model.plot(x_test[1].reshape((-1, 1)), predictions.item(2))
        self.assertTrue(os.path.isfile(filename + '.png'))

    def test_plot_plotname(self):
        _, trained_model = self._train_load_model()

        x_test = dataset.x_test

        trained_model_predictions_onehot = trained_model.predict(x_test)
        predictions = libs.one_class_prediction(trained_model_predictions_onehot)

        plot_name = 'neurocat_plot'
        trained_model.plot(x_test[1], predictions.item(2), plot_name=plot_name)

        filename = os.path.join('figures', plot_name)
        self.assertTrue(os.path.isfile(filename + '.png'))

    def test_adversarial_attack(self):
        model, trained_model = self._train_load_model(50)

        x_test = dataset.x_test[0]
        y_test = dataset.y_test[0]

        adv_image = adversarial_attack.fast_gradient_sign_method(trained_model, x_test, y_test)

        img_prediction_onehot = libs.one_class_prediction(trained_model.predict(x_test))
        adv_image_prediction_onehot = libs.one_class_prediction(trained_model.predict(adv_image))
        self.assertNotEquals(img_prediction_onehot, adv_image_prediction_onehot)


if __name__ == '__main__':
    unittest.main()
