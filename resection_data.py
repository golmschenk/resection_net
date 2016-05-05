"""
Code for managing the resectioning data.
"""
from math import pi, acos
import h5py
import numpy as np
import tensorflow as tf

from go_data import GoData


class ResectionData(GoData):
    """
    A class for managing the resectioning data.
    """

    def convert_mat_to_tfrecord(self, mat_file_path):
        """
        Converts the mat file data into a TFRecords file.
        Overrides the GoData method, because only images should be cropped.

        :param mat_file_path: The path to mat file to convert.
        :type mat_file_path: str
        """
        mat_data = h5py.File(mat_file_path, 'r')
        uncropped_images = self.convert_mat_data_to_numpy_array(mat_data, 'images')
        self.images = self.crop_data(uncropped_images)
        acceleration_vectors = self.convert_mat_data_to_numpy_array(mat_data, 'accelData')[:, :3]
        gravity_vectors = acceleration_vectors * -1  # The acceleration is in the up direction.
        self.labels = np.zeros((self.images.shape[0], 2))
        for index, gravity_vector in enumerate(gravity_vectors):
            normalized_gravity_vector = tuple(self.normalize_vector(gravity_vector))
            self.labels[index][0] = self.attain_pitch_from_gravity_vector(normalized_gravity_vector)
            self.labels[index][1] = self.attain_roll_from_gravity_vector(normalized_gravity_vector)
        self.shrink()
        self.convert_to_tfrecord()

    def shrink(self):
        """
        Rebins the data arrays into the specified data size.
        Overrides the GoData method, because only images should be resized.
        """
        self.images = self.shrink_array_with_rebinning(self.images)

    def reshape_decoded_label(self, flat_label):
        """
        Reshapes the label decoded from the TF record.
        Overrides the GoData method, because the labels are of a different shape than the default.

        :param flat_label: The flat label from the decoded TF record.
        :type flat_label: tf.Tensor
        :return: The reshaped label.
        :rtype: tf.Tensor
        """
        return tf.reshape(flat_label, [2])

    @staticmethod
    def attain_pitch_from_gravity_vector(gravity_vector):
        """
        Determines the pitch angle given the normalized gravity vector.

        :param gravity_vector: The normalized gravity vector.
        :type gravity_vector: (int, int, int)
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
        :type gravity_vector: (int, int, int)
        :return: The roll angle.
        :rtype: float
        """
        xy_vector = list(gravity_vector)
        xy_vector[2] = 0
        xy_vector = self.normalize_vector(xy_vector)

        x = xy_vector[0]
        roll = -(pi / 2 - acos(x))
        return roll


if __name__ == '__main__':
    data = ResectionData()
    data.convert_mat_to_tfrecord('data/nyu_depth_v2_labeled.mat')
