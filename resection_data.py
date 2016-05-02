"""
Code for managing the resectioning data.
"""
from math import pi, acos
import numpy as np

from go_data import GoData


class ResectionData(GoData):
    """
    A class for managing the resectioning data.
    """
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
        xy_vector = gravity_vector
        xy_vector[2] = 0
        xy_vector = self.normalize_vector(xy_vector)

        x = xy_vector[0]
        roll = -(pi / 2 - acos(x))
        return roll


if __name__ == '__main__':
    data = ResectionData()
    data.convert_mat_to_tfrecord('data/nyud_micro.mat')
