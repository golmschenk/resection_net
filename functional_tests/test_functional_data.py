"""
Tests the data preparation scripts.
"""
import os
import math
import numpy as np
import tensorflow as tf
from pytest import fail

from go_data import GoData


class TestFunctionalData:
    def test_can_convert_from_mat_file_to_numpy_files(self):
        # Prepare paths.
        images_numpy_file_path = os.path.join('functional_tests', 'test_data', 'images_nyud_micro.npy')
        depths_numpy_file_path = os.path.join('functional_tests', 'test_data', 'depths_nyud_micro.npy')
        mat_file_path = os.path.join('functional_tests', 'test_data', 'nyud_micro.mat')

        # Run the conversion script.
        GoData().convert_mat_file_to_numpy_file(mat_file_path)

        # Check that the files are created.
        assert os.path.isfile(images_numpy_file_path)
        assert os.path.isfile(depths_numpy_file_path)

        # Check that magic values are correct when the data is reloaded from numpy files.
        images = np.load(images_numpy_file_path)
        assert images[5, 10, 10, 1] == 91
        depths = np.load(depths_numpy_file_path)
        assert math.isclose(depths[5, 10, 10], 3.75686, abs_tol=0.001)

        # Clean up.
        remove_file_if_exists(images_numpy_file_path)
        remove_file_if_exists(depths_numpy_file_path)


def remove_file_if_exists(file_path):
    try:
        os.remove(file_path)
    except OSError:
        pass
