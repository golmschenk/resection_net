"""
Code for managing the resectioning data.
"""
from math import pi, acos
import h5py
import numpy as np

from go_data import GoData


class ResectionData(GoData):
    """
    A class for managing the resectioning data.
    """
    def __init__(self):
        super().__init__()

        self.height = self.original_height
        self.width = self.original_width
        self.label_shape = [2]

        self.train_size = 'all'
        self.validation_size = 0

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
            acceleration_vectors = self.convert_mat_data_to_numpy_array(mat_data, 'accelData')[:, :3]
            gravity_vectors = np.multiply(acceleration_vectors, -1)  # The acceleration is in the up direction.
            labels = np.zeros((len(gravity_vectors), 2), dtype=np.float32)
            for index, gravity_vector in enumerate(gravity_vectors):
                normalized_gravity_vector = tuple(self.normalize_vector(gravity_vector))
                labels[index][0] = self.attain_pitch_from_gravity_vector(normalized_gravity_vector)
                labels[index][1] = self.attain_roll_from_gravity_vector(normalized_gravity_vector)
            if self.images is None:
                self.images = images
                self.labels = labels
            else:
                self.images = np.concatenate((self.images, images))
                self.labels = np.concatenate((self.labels, labels))

    def preprocess(self):
        """
        Preprocesses the data.
        Should be overwritten by subclasses.
        """
        print('Shrinking the data...')
        self.shrink()
        print('Shuffling the data...')
        self.shuffle()

    def shrink(self):
        """
        Rebins the data arrays into the specified data size.
        Overrides the GoData method, because only images should be resized.
        """
        self.images = self.shrink_array_with_rebinning(self.images)

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
    data.generate_tfrecords()
