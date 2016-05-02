"""
Code for managing the resectioning data.
"""
from math import pi, acos

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


if __name__ == '__main__':
    data = ResectionData()
    data.convert_mat_to_tfrecord('data/nyud_micro.mat')
