"""
Code related to the ResectionNet.
"""
import tensorflow as tf

from resection_data import ResectionData
from go_net import GoNet
from interface import Interface
from convenience import weight_variable, bias_variable, leaky_relu, conv2d, size_from_stride_two


class ResectionNet(GoNet):
    """
    A neural network class to estimate camera parameters from 2D images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = ResectionData()
        self.epoch_limit = None
        self.step_summary_name = "Loss"
        self.image_summary_on = False

        self.batch_size = 50
        self.initial_learning_rate = 0.00001

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
        return self.create_deep_with_dropout_inference_op(images)

    def create_linear_classifier_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using only a linear classifier.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        pixel_count = self.data.image_height * self.data.image_width
        flat_images = tf.reshape(images, [-1, pixel_count * self.data.image_depth])
        weights = weight_variable([pixel_count * self.data.image_depth, 2], stddev=0.001)
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
        pixel_count = self.data.image_height * self.data.image_width
        flat_images = tf.reshape(images, [-1, pixel_count * self.data.image_depth])
        weights = weight_variable([pixel_count * self.data.image_depth, 64], stddev=0.001)
        biases = bias_variable([64], constant=0.001)

        flat_hypothesis = tf.matmul(flat_images, weights) + biases

        weights = weight_variable([64, 2], stddev=0.001)
        biases = bias_variable([2], constant=0.001)
        flat_predicted_labels = tf.matmul(flat_hypothesis, weights) + biases

        predicted_labels = tf.reshape(flat_predicted_labels, [-1, 2])
        return predicted_labels

    def create_deep_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using a deep convolution net.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        with tf.name_scope('conv1'):
            w_conv = weight_variable([3, 3, 3, 16])
            b_conv = bias_variable([16])

            h_conv = leaky_relu(conv2d(images, w_conv, strides=[1, 2, 2, 1]) + b_conv)

        with tf.name_scope('conv2'):
            w_conv = weight_variable([3, 3, 16, 32])
            b_conv = bias_variable([32])

            h_conv = leaky_relu(conv2d(h_conv, w_conv, strides=[1, 2, 2, 1]) + b_conv)

        with tf.name_scope('conv3'):
            w_conv = weight_variable([3, 3, 32, 64])
            b_conv = bias_variable([64])

            h_conv = leaky_relu(conv2d(h_conv, w_conv, strides=[1, 2, 2, 1]) + b_conv)

        with tf.name_scope('conv4'):
            w_conv = weight_variable([3, 3, 64, 128])
            b_conv = bias_variable([128])

            h_conv = leaky_relu(conv2d(h_conv, w_conv, strides=[1, 2, 2, 1]) + b_conv)

        with tf.name_scope('conv5'):
            w_conv = weight_variable([3, 3, 128, 256])
            b_conv = bias_variable([256])

            h_conv = leaky_relu(conv2d(h_conv, w_conv, strides=[1, 2, 2, 1]) + b_conv)

        with tf.name_scope('conv6'):
            w_conv = weight_variable([10, 10, 256, 256])
            b_conv = bias_variable([256])

            h_conv = leaky_relu(conv2d(h_conv, w_conv, strides=[1, 2, 2, 1]) + b_conv)

        with tf.name_scope('fc1'):
            fc0_size = size_from_stride_two(self.data.image_height, iterations=6) * size_from_stride_two(
                self.data.image_width, iterations=6) * 256
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
        with tf.name_scope('conv1'):
            w_conv = weight_variable([3, 3, 3, 16])
            b_conv = bias_variable([16])

            h_conv = leaky_relu(conv2d(images, w_conv, strides=[1, 2, 2, 1]) + b_conv)

        with tf.name_scope('conv2'):
            w_conv = weight_variable([3, 3, 16, 32])
            b_conv = bias_variable([32])

            h_conv = leaky_relu(conv2d(h_conv, w_conv, strides=[1, 2, 2, 1]) + b_conv)

        with tf.name_scope('conv3'):
            w_conv = weight_variable([3, 3, 32, 64])
            b_conv = bias_variable([64])

            h_conv = leaky_relu(conv2d(h_conv, w_conv, strides=[1, 2, 2, 1]) + b_conv)
            h_conv_drop = tf.nn.dropout(h_conv, self.dropout_keep_probability_tensor)

        with tf.name_scope('conv4'):
            w_conv = weight_variable([3, 3, 64, 128])
            b_conv = bias_variable([128])

            h_conv = leaky_relu(conv2d(h_conv_drop, w_conv, strides=[1, 2, 2, 1]) + b_conv)
            h_conv_drop = tf.nn.dropout(h_conv, self.dropout_keep_probability_tensor)

        with tf.name_scope('conv5'):
            w_conv = weight_variable([3, 3, 128, 256])
            b_conv = bias_variable([256])

            h_conv = leaky_relu(conv2d(h_conv_drop, w_conv, strides=[1, 2, 2, 1]) + b_conv)
            h_conv_drop = tf.nn.dropout(h_conv, self.dropout_keep_probability_tensor)

        with tf.name_scope('conv6'):
            w_conv = weight_variable([10, 10, 256, 256])
            b_conv = bias_variable([256])

            h_conv = leaky_relu(conv2d(h_conv_drop, w_conv, strides=[1, 2, 2, 1]) + b_conv)
            h_conv_drop = tf.nn.dropout(h_conv, self.dropout_keep_probability_tensor)

        with tf.name_scope('fc1'):
            fc0_size = size_from_stride_two(self.data.image_height, iterations=6) * size_from_stride_two(
                self.data.image_width, iterations=6) * 256
            fc1_size = 2
            h_fc = tf.reshape(h_conv_drop, [-1, fc0_size])
            w_fc = weight_variable([fc0_size, fc1_size])
            b_fc = bias_variable([fc1_size])

            predicted_labels = leaky_relu(tf.matmul(h_fc, w_fc) + b_fc)

        return predicted_labels


if __name__ == '__main__':
    interface = Interface(network_class=ResectionNet)
    interface.train()
