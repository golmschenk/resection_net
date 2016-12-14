"""
Code related to the ResectionNet.
"""
import numpy as np
import tensorflow as tf
import math

from tensorflow.contrib.layers import batch_norm, max_pool2d, fully_connected, flatten

from resection_data import ResectionData
from gonet.net import Net
from gonet.interface import Interface
from gonet.convenience import weight_variable, bias_variable, leaky_relu, size_from_stride_two, conv_layer

from settings import Settings


class ResectionNet(Net):
    """
    A neural network class to estimate camera parameters from 2D images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(settings=Settings(), *args, **kwargs)

        self.data = ResectionData()

        self.step_summary_name = "Loss"
        self.image_summary_on = False
        self.test_labels = None

    def create_loss_tensor(self, predicted_labels, labels):
        """
        Create the loss op and add it to the graph.

        :param predicted_labels: The labels predicted by the graph.
        :type predicted_labels: tf.Tensor
        :param labels: The ground truth labels.
        :type labels: tf.Tensor
        :return: The loss tensor.
        :rtype: tf.Tensor
        """
        squared_difference = tf.square(predicted_labels - labels)
        tf.scalar_summary("Worst squared difference", tf.reduce_max(squared_difference))
        return squared_difference

    def create_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        inference_op = self.create_striding_hermes_inference_op(images)
        return tf.identity(inference_op, name='inference_op')

    def create_linear_classifier_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using only a linear classifier.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        pixel_count = self.settings.image_height * self.settings.image_width
        flat_images = tf.reshape(images, [-1, pixel_count * self.settings.image_depth])
        weights = weight_variable([pixel_count * self.settings.image_depth, 2], stddev=0.001)
        biases = bias_variable([2], constant=0.001)

        flat_predicted_labels = tf.matmul(flat_images, weights) + biases
        predicted_labels = tf.reshape(flat_predicted_labels, [-1, 2])
        return predicted_labels

    def create_two_layer_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using 2 fully connected layers.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        pixel_count = self.settings.image_height * self.settings.image_width
        flat_images = tf.reshape(images, [-1, pixel_count * self.settings.image_depth])
        weights = weight_variable([pixel_count * self.settings.image_depth, 64], stddev=0.001)
        biases = bias_variable([64], constant=0.001)

        flat_hypothesis = tf.matmul(flat_images, weights) + biases

        weights = weight_variable([64, 2], stddev=0.001)
        biases = bias_variable([2], constant=0.001)
        flat_predicted_labels = tf.matmul(flat_hypothesis, weights) + biases

        predicted_labels = tf.reshape(flat_predicted_labels, [-1, 2])
        return predicted_labels

    def create_striding_hermes_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using the mercury modules.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        module1_output = self.mercury_module('module1', images, 3, 8, 3)
        max_pool1_output = max_pool2d(module1_output, kernel_size=3, stride=2, padding='VALID')
        module2_output = self.mercury_module('module2', max_pool1_output, 10, 20, 10)
        max_pool2_output = max_pool2d(module2_output, kernel_size=3, stride=2, padding='VALID')
        module3_output = self.mercury_module('module3', max_pool2_output, 20, 40, 20)
        max_pool3_output = max_pool2d(module3_output, kernel_size=3, stride=2, padding='VALID')
        module4_output = self.mercury_module('module4', max_pool3_output, 30, 60, 30)
        max_pool4_output = max_pool2d(module4_output, kernel_size=3, stride=2, padding='VALID')
        module5_output = self.mercury_module('module5', max_pool4_output, 50, 100, 50, dropout_on=True)
        max_pool5_output = max_pool2d(module5_output, kernel_size=3, stride=2, padding='VALID')
        module6_output = self.mercury_module('module6', max_pool5_output, 75, 150, 75, dropout_on=True)
        max_pool6_output = max_pool2d(module6_output, kernel_size=3, stride=2, padding='VALID')
        predicted_labels = fully_connected(flatten(max_pool6_output), 2, activation_fn=None)
        return predicted_labels

    def create_striding_gaea_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using the terra modules

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        module1_output = self.terra_module('module1', images, 16)
        max_pool1_output = max_pool2d(module1_output, kernel_size=3, stride=2, padding='VALID')
        module2_output = self.terra_module('module1', max_pool1_output, 32)
        max_pool2_output = max_pool2d(module2_output, kernel_size=3, stride=2, padding='VALID')
        module3_output = self.terra_module('module1', max_pool2_output, 64)
        max_pool3_output = max_pool2d(module3_output, kernel_size=3, stride=2, padding='VALID')
        module4_output = self.terra_module('module1', max_pool3_output, 128, dropout_on=True)
        max_pool4_output = max_pool2d(module4_output, kernel_size=3, stride=2, padding='VALID')
        module5_output = self.terra_module('module1', max_pool4_output, 256, dropout_on=True)
        max_pool5_output = max_pool2d(module5_output, kernel_size=3, stride=2, padding='VALID')
        module6_output = self.terra_module('module1', max_pool5_output, 256, dropout_on=True)
        max_pool6_output = max_pool2d(module6_output, kernel_size=3, stride=2, padding='VALID')
        predicted_labels = fully_connected(flatten(max_pool6_output), 2, activation_fn=None)
        return predicted_labels

    def create_deep_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using a deep convolution net.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        h_conv = conv_layer('conv1', images, 3, 16, strides=(1, 2, 2, 1))
        h_conv = conv_layer('conv2', h_conv, 16, 32, strides=(1, 2, 2, 1))
        h_conv = conv_layer('conv3', h_conv, 32, 64, strides=(1, 2, 2, 1))
        h_conv = conv_layer('conv4', h_conv, 64, 128, strides=(1, 2, 2, 1))
        h_conv = conv_layer('conv5', h_conv, 128, 256, strides=(1, 2, 2, 1))
        h_conv = conv_layer('conv6', h_conv, 256, 256, conv_height=10, conv_width=10, strides=(1, 2, 2, 1))

        with tf.name_scope('fc1'):
            fc0_size = size_from_stride_two(self.settings.image_height, iterations=6) * size_from_stride_two(
                self.settings.image_width, iterations=6) * 256
            fc1_size = 2
            h_fc = tf.reshape(h_conv, [-1, fc0_size])
            w_fc = weight_variable([fc0_size, fc1_size])
            b_fc = bias_variable([fc1_size])

            predicted_labels = leaky_relu(tf.matmul(h_fc, w_fc) + b_fc)

        return predicted_labels

    def create_deep_with_dropout_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using a deep convolution net.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """

        h_conv = conv_layer('conv1', images, 3, 16, strides=(1, 2, 2, 1), histogram_summary=True)
        h_conv = batch_norm(h_conv)
        h_conv = conv_layer('conv2', h_conv, 16, 32, strides=(1, 2, 2, 1), histogram_summary=True)
        h_conv = batch_norm(h_conv)
        h_conv = conv_layer('conv3', h_conv, 32, 64, strides=(1, 2, 2, 1), histogram_summary=True)
        h_conv = tf.nn.dropout(h_conv, self.dropout_keep_probability_tensor)
        h_conv = batch_norm(h_conv)
        h_conv = conv_layer('conv4', h_conv, 64, 128, strides=(1, 2, 2, 1), histogram_summary=True)
        h_conv = tf.nn.dropout(h_conv, self.dropout_keep_probability_tensor)
        h_conv = batch_norm(h_conv)
        h_conv = conv_layer('conv5', h_conv, 128, 256, strides=(1, 2, 2, 1), histogram_summary=True)
        h_conv = tf.nn.dropout(h_conv, self.dropout_keep_probability_tensor)
        h_conv = batch_norm(h_conv)
        h_conv = conv_layer('conv6', h_conv, 256, 256, conv_height=10, conv_width=10, strides=(1, 2, 2, 1),
                            histogram_summary=True)
        h_conv_drop = tf.nn.dropout(h_conv, self.dropout_keep_probability_tensor)

        with tf.name_scope('fc1'):
            fc0_size = size_from_stride_two(self.settings.image_height, iterations=6) * size_from_stride_two(
                self.settings.image_width, iterations=6) * 256
            fc1_size = 2
            h_fc = tf.reshape(h_conv_drop, [-1, fc0_size])
            w_fc = weight_variable([fc0_size, fc1_size])
            b_fc = bias_variable([fc1_size])

            predicted_labels = leaky_relu(tf.matmul(h_fc, w_fc) + b_fc)

        return predicted_labels

    def test_run_preloop(self):
        """
        The code run before the test loop. Mostly for setting up things that will be used within the loop.
        """
        self.test_labels = np.ndarray(shape=[0, 2], dtype=np.float32)
        self.predicted_test_labels = np.ndarray(shape=[0, 2], dtype=np.float32)

    def test_run_loop_step(self):
        """
        The code that will be used during the each iteration of the test loop (excluding the step incrementation).
        """
        predicted_labels_tensor = self.session.graph.get_tensor_by_name('inference_op:0')
        labels_tensor = self.session.graph.get_tensor_by_name('labels_input_op:0')
        predicted_labels_batch, labels_batch = self.session.run(
            [predicted_labels_tensor, labels_tensor],
            feed_dict={**self.default_feed_dictionary, self.dropout_keep_probability_tensor: 1.0}
        )
        self.test_labels = np.concatenate((self.test_labels, labels_batch))
        self.predicted_test_labels = np.concatenate((self.predicted_test_labels, predicted_labels_batch))
        print('{image_count} images processed.'.format(image_count=(self.global_step + 1) * self.settings.batch_size))

    def test_run_postloop(self):
        """
        The code that will be run once the inference test loop is finished. Mostly for saving data or statistics.
        """
        # Take into account that 1 was added to all label values.
        self.predicted_test_labels -= 1.0
        self.test_labels -= 1.0

        absolute_difference = np.abs(self.predicted_test_labels - self.test_labels)
        squared_difference = np.square(absolute_difference)
        axis_mean_labels = np.mean(self.test_labels, axis=0)
        absolute_deviation = np.abs(self.test_labels - axis_mean_labels)
        mean_difference = np.mean(absolute_difference)
        mean_squared_difference = np.mean(squared_difference)
        axis_mean_difference = np.mean(absolute_difference, axis=0)
        axis_mean_squared_difference = np.mean(squared_difference, axis=0)
        axis_labels_standard_deviation = np.std(self.test_labels, axis=0)
        axis_mean_absolute_deviation = np.mean(absolute_deviation, axis=0)

        print('Radians')
        print('Combined mean absolute difference: {}'.format(mean_difference))
        print('Combined mean squared difference: {}'.format(mean_squared_difference))
        print('Pitch mean absolute difference: {}'.format(axis_mean_difference[0]))
        print('Pitch mean squared difference: {}'.format(axis_mean_squared_difference[0]))
        print('Roll mean absolute difference: {}'.format(axis_mean_difference[1]))
        print('Roll mean squared difference: {}'.format(axis_mean_squared_difference[1]))
        print('Ground truth pitch mean: {}'.format(axis_mean_labels[0]))
        print('Ground truth roll mean: {}'.format(axis_mean_labels[1]))
        print('Ground truth combined standard deviation: {}'.format(np.std(self.test_labels)))
        print('Ground truth pitch standard deviation: {}'.format(axis_labels_standard_deviation[0]))
        print('Ground truth roll standard deviation: {}'.format(axis_labels_standard_deviation[1]))
        print('Ground truth combined mean absolute deviation: {}'.format(np.mean(absolute_deviation)))
        print('Ground truth pitch mean absolute deviation: {}'.format(axis_mean_absolute_deviation[0]))
        print('Ground truth roll mean absolute deviation: {}'.format(axis_mean_absolute_deviation[1]))
        print('Degrees')
        print('Ground truth pitch mean: {}'.format(math.degrees(axis_mean_labels[0])))
        print('Ground truth roll mean: {}'.format(math.degrees(axis_mean_labels[1])))
        print('Ground truth combined standard deviation: {}'.format(math.degrees(np.std(self.test_labels))))
        print('Combined mean squared difference: {}'.format(math.degrees(mean_squared_difference)))
        print('Ground truth pitch standard deviation: {}'.format(math.degrees(axis_labels_standard_deviation[0])))
        print('Pitch mean squared difference: {}'.format(math.degrees(axis_mean_squared_difference[0])))
        print('Ground truth roll standard deviation: {}'.format(math.degrees(axis_labels_standard_deviation[1])))
        print('Roll mean squared difference: {}'.format(math.degrees(axis_mean_squared_difference[1])))
        print('Ground truth combined mean absolute deviation: {}'.format(math.degrees(np.mean(absolute_deviation))))
        print('Combined mean absolute difference: {}'.format(math.degrees(mean_difference)))
        print('Ground truth pitch mean absolute deviation: {}'.format(math.degrees(axis_mean_absolute_deviation[0])))
        print('Pitch mean absolute difference: {}'.format(math.degrees(axis_mean_difference[0])))
        print('Ground truth roll mean absolute deviation: {}'.format(math.degrees(axis_mean_absolute_deviation[1])))
        print('Roll mean absolute difference: {}'.format(math.degrees(axis_mean_difference[1])))


if __name__ == '__main__':
    interface = Interface(network_class=ResectionNet)
    interface.run()
