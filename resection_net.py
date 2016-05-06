"""
Code related to the ResectionNet.
"""
import tensorflow as tf

from resection_data import ResectionData
from go_net import GoNet
from interface import Interface
from convenience import weight_variable, bias_variable, leaky_relu, conv2d


class ResectionNet(GoNet):
    """
    A neural network class to estimate camera parameters from 2D images.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = ResectionData()
        self.step_summary_name = "Loss"
        self.image_summary_on = False

    def create_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        return self.create_linear_classifier_inference_op(images)

    def create_linear_classifier_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using only a linear classifier.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        pixel_count = self.data.height * self.data.width
        flat_images = tf.reshape(images, [-1, pixel_count * self.data.channels])
        weights = weight_variable([pixel_count * self.data.channels, 2], stddev=0.001)
        biases = bias_variable([2], constant=0.001)

        flat_predicted_labels = tf.matmul(flat_images, weights) + biases
        predicted_labels = tf.reshape(flat_predicted_labels, [-1, 2])
        return predicted_labels

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
        return self.create_absolute_differences_tensor(predicted_labels, labels)


if __name__ == '__main__':
    interface = Interface(network_class=ResectionNet)
    interface.train()
