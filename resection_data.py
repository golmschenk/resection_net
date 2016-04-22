"""
Code for managing the resectioning data.
"""
from go_data import GoData


class ResectionData(GoData):
    """
    A class for managing the resectioning data.
    """
    pass


if __name__ == '__main__':
    data = ResectionData()
    data.convert_mat_to_tfrecord('data/nyud_micro.mat')
