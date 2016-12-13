"""
Code for managing the resectioning data.
"""
from math import pi, acos
import h5py
import numpy as np
import tensorflow as tf

from gonet.data import Data

from ground_truth import GroundTruth
from settings import Settings


class ResectionData(Data):
    """
    A class for managing the resectioning data.
    """
    def __init__(self):
        super().__init__(settings=Settings())

    def import_mat_file(self, mat_path):
        """
        Imports a Matlab mat file into the data images and labels (concatenating the arrays if they already exists).
        Overrides the GoData method, because only images should be cropped and processing must be done for the labels.

        :param mat_path: The path to the mat file to import.
        :type mat_path: str
        """
        with h5py.File(mat_path, 'r') as mat_data:
            uncropped_images = self.convert_mat_data_to_numpy_array(mat_data, 'images')
            images = self.crop_data(uncropped_images)
            acceleration_data = self.convert_mat_data_to_numpy_array(mat_data, 'accelData')
            if acceleration_data.shape[0] == 4:
                assert acceleration_data.shape[1] != 4
                acceleration_data = acceleration_data.transpose()
            acceleration_vectors = acceleration_data[:, :3]
            gravity_vectors = np.multiply(acceleration_vectors, -1)  # The acceleration is in the up direction.
            labels = np.zeros((len(gravity_vectors), 2), dtype=np.float32)
            for index, gravity_vector in enumerate(gravity_vectors):
                normalized_gravity_vector = self.normalize_vector(gravity_vector)
                labels[index][0] = self.attain_pitch_from_gravity_vector(normalized_gravity_vector)
                labels[index][1] = self.attain_roll_from_gravity_vector(normalized_gravity_vector)
            self.images = images
            self.labels = labels

    @staticmethod
    def attain_pitch_from_gravity_vector(gravity_vector):
        """
        Determines the pitch angle given the normalized gravity vector.

        :param gravity_vector: The normalized gravity vector.
        :type gravity_vector: list[int]
        :return: The pitch angle.
        :rtype: float
        """
        z = gravity_vector[2]
        pitch = -(pi/2 - acos(z))
        return pitch

    @staticmethod
    def normalize_vector(vector):
        """
        Normalize a given vector.

        :param vector: The vector to normalize.
        :type vector: list[int]
        :return: The normalized vector.
        :rtype: list[int]
        """
        return vector / np.linalg.norm(vector)

    def attain_roll_from_gravity_vector(self, gravity_vector):
        """
        Determines the roll angle given the normalized gravity vector.

        :param gravity_vector: The normalized gravity vector.
        :type gravity_vector: list[int]
        :return: The roll angle.
        :rtype: float
        """
        xy_vector = list(gravity_vector)
        xy_vector[2] = 0
        xy_vector = self.normalize_vector(xy_vector)

        x = xy_vector[0]
        roll = -(pi / 2 - acos(x))
        return roll

    def preaugmentation_preprocess(self, image, label):
        """
        Preprocesses the image and label to be in the correct format for training.
        Overrides GoNet method.

        :param image: The image to be processed.
        :type image: tf.Tensor
        :param label: The label to be processed.
        :type label: tf.Tensor
        :return: The processed image and label.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        image = tf.image.resize_images(image, [self.settings.image_height, self.settings.image_width])
        label = tf.reshape(label, [self.settings.label_height]) + 1.0
        return image, label

    @staticmethod
    def horizontally_flip_label(label):
        """
        Changes the roll to be the negative of its current value.
        Overrides GoNet method.

        :param label: The label to be "flipped".
        :type label: tf.Tensor
        :return: The "flipped" label.
        :rtype: tf.Tensor
        """
        return tf.mul(label, [1.0, -1.0])

    @staticmethod
    def extract_pitch_and_roll_from_sun_rgbd_extrinsics_text_file(file_path):
        """
        Gets the pitch and roll from a SUN RGB-D extrinsics file.

        :param file_path: The path to the extrinsics file.
        :type file_path: str
        :return: The pitch and the roll.
        :rtype: (float, float)
        """
        extrinsics_array = np.transpose(np.loadtxt(file_path, dtype=np.float32))
        ground_truth = GroundTruth()
        ground_truth.r1 = extrinsics_array[0]
        ground_truth.r2 = extrinsics_array[1]
        ground_truth.r3 = extrinsics_array[2]
        return -ground_truth.pitch(), -ground_truth.roll()


if __name__ == '__main__':
    data = ResectionData()
    data.generate_all_tfrecords()
