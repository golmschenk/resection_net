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
        z = gravity_vector[2]
        pitch = -(pi/2 - acos(z))
        return pitch

    @staticmethod
    def normalize_vector(vector):
        return vector / np.linalg.norm(vector)

    def attain_roll_from_gravity_vector(self, gravity_vector):
        xy_vector = gravity_vector
        xy_vector[2] = 0
        xy_vector = self.normalize_vector(xy_vector)

        x = xy_vector[0]
        roll = -(pi / 2 - acos(x))
        return roll


if __name__ == '__main__':
    data = ResectionData()
    data.convert_mat_to_tfrecord('data/nyud_micro.mat')
