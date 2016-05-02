"""
A module for tests related to the ResectionData class.
"""
from math import pi, isclose

from resection_data import ResectionData


class TestResectionData:
    """
    A test suite for the ResectionData class.
    """
    def test_calculation_of_pitch_from_gravity_vector(self):
        gravity_vector0 = [0.0, -1.0, 0.0]
        gravity_vector1 = [0.0, -0.707, 0.707]
        gravity_vector2 = [0.0, -0.707, -0.707]
        gravity_vector3 = [0.0, 0.0, 1.0]
        gravity_vector4 = [0.577, 0.577, 0.577]

        pitch0 = ResectionData().attain_pitch_from_gravity_vector(gravity_vector0)
        pitch1 = ResectionData().attain_pitch_from_gravity_vector(gravity_vector1)
        pitch2 = ResectionData().attain_pitch_from_gravity_vector(gravity_vector2)
        pitch3 = ResectionData().attain_pitch_from_gravity_vector(gravity_vector3)
        pitch4 = ResectionData().attain_pitch_from_gravity_vector(gravity_vector4)

        assert isclose(pitch0, 0, rel_tol=0.001)
        assert isclose(pitch1, -pi/4, rel_tol=0.001)
        assert isclose(pitch2, pi/4, rel_tol=0.001)
        assert isclose(pitch3, -pi/2, rel_tol=0.001)
        assert isclose(pitch4, -0.615, rel_tol=0.001)
