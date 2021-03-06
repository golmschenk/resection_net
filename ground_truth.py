"""
Code for generating ground truth data to check the accelerometer data accuracy.
"""
from math import atan2, pi

import numpy as np
from collections import namedtuple
from xml.etree import ElementTree

import sys

Point = namedtuple('Point', ['x', 'y'])
VanishingPoint = namedtuple('VanishingPoint', ['x', 'y', 'axis'])
LineSegment = namedtuple('Line', ['start', 'end', 'axis'])

kinect_factory_calibration = np.array([[5.2161910696979987e+02, 0., 3.1755491910920682e+02],
                                       [0., 5.2132946256749767e+02, 2.5921654718027673e+02],
                                       [0., 0., 1.]], dtype=np.float32)


class GroundTruth:
    """
    A class to generate ground truth to check the accelerometer data accuracy.
    """

    def __init__(self):
        self.r1 = None
        self.r2 = None
        self.r3 = None

    @staticmethod
    def extract_line_segments_from_label_me_xml(xml_file_name):
        """
        Extracts the list of line segments from a LabelMe xml file.

        :param xml_file_name: The name of the xml file.
        :type xml_file_name: str
        :return: The list of lines segments.
        :rtype: list[LineSegment]
        """
        tree = ElementTree.parse(xml_file_name)
        root = tree.getroot()
        line_segments = []
        for object_entry in root.findall('object'):
            polygon = object_entry.find('polygon')
            points = polygon.findall('pt')
            if len(points) != 2 or object_entry.find('attributes').text not in ('x', 'y', 'z'):
                continue
            point0 = Point(x=int(points[0][0].text), y=int(points[0][1].text))
            point1 = Point(x=int(points[1][0].text), y=int(points[1][1].text))
            axis = object_entry.find('attributes').text
            line_segment = LineSegment(start=point0, end=point1, axis=axis)
            line_segments.append(line_segment)
        return line_segments

    @staticmethod
    def attain_line_intersection(line_segment1, line_segment2):
        """
        Gives where two line segments intersect (if they were infinitely long).

        :param line_segment1: The first line segment.
        :type line_segment1: LineSegment
        :param line_segment2: The second line segment.
        :type line_segment2: LineSegment
        :return: The point of the intersection
        :rtype:
        """
        x_diff = Point(line_segment1.start.x - line_segment1.end.x, line_segment2.start.x - line_segment2.end.x)
        y_diff = Point(line_segment1.start.y - line_segment1.end.y, line_segment2.start.y - line_segment2.end.y)

        def determinant(a, b):
            """
            Gets the determininant of the two points.

            :param a: The first point.
            :type a: Point
            :param b: The second point.
            :type b: Point
            :return: The determinant
            :rtype: float
            """
            return a.x * b.y - a.y * b.x

        div = determinant(x_diff, y_diff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = Point(determinant(line_segment1.start, line_segment1.end),
                  determinant(line_segment2.start, line_segment2.end))
        x = determinant(d, x_diff) / div
        y = determinant(d, y_diff) / div
        return Point(x, y)

    @staticmethod
    def attain_line_segment_pairs_from_line_segments(line_segments):
        """
        Gets the pairs of line segments from the list of line_segments.

        :param line_segments: The list of line segments to sort through.
        :type line_segments: list[LineSegments]
        :return: The pairs of line segments.
        :rtype: list[(LineSegment, LineSegment)]
        """
        x_pair = [line_segment for line_segment in line_segments if line_segment.axis == 'x']
        y_pair = [line_segment for line_segment in line_segments if line_segment.axis == 'y']
        z_pair = [line_segment for line_segment in line_segments if line_segment.axis == 'z']
        pairs = []
        if len(x_pair) == 2:
            pairs.append(x_pair)
        if len(y_pair) == 2:
            pairs.append(y_pair)
        if len(z_pair) == 2:
            pairs.append(z_pair)
        if len(pairs) == 2:
            return pairs
        else:
            sys.exit("Found {} segment pairs. Should have found 2.".format(len(pairs)))

    def attain_vanishing_points_from_line_segment_pairs(self, line_segment_pairs):
        """
        Gets the vanishing points from the line segment pairs.

        :param line_segment_pairs: The line segment pairs.
        :type line_segment_pairs: list[(LineSegment, LineSegment)]
        :return: The vanishing points.
        :rtype: list[VanishingPoint]
        """
        vanishing_points = []
        for line_segment_pair in line_segment_pairs:
            intersection_point = self.attain_line_intersection(line_segment_pair[0], line_segment_pair[1])
            vanishing_point = VanishingPoint(intersection_point[0], intersection_point[1], line_segment_pair[0].axis)
            vanishing_points.append(vanishing_point)
        return vanishing_points

    def obtain_rotation_column(self, vanishing_point):
        """
        Extracts the rotation information from the vanishing point and adds it to the object.

        :param vanishing_point: The vanishing point to process.
        :type vanishing_point: VanishingPoint
        """
        vanishing_point_column = np.array([[vanishing_point[0]], [vanishing_point[1]], [1.0]], dtype=np.float32)
        unnormalized_r = np.dot(np.linalg.inv(kinect_factory_calibration), vanishing_point_column)
        r = unnormalized_r / np.linalg.norm(unnormalized_r)
        if vanishing_point[2] == 'x':
            self.r1 = r
        elif vanishing_point[2] == 'y':
            self.r2 = r
        elif vanishing_point[2] == 'z':
            self.r3 = r

    def obtain_missing_rotation_column(self):
        """
        Given that the other two rotation columns are already calculated, calculate the last one.
        """
        if self.r1 is None:
            self.r1 = np.cross(self.r2, self.r3, axis=0)
        elif self.r2 is None:
            self.r2 = np.cross(self.r3, self.r1, axis=0)
        elif self.r3 is None:
            self.r3 = np.cross(self.r1, self.r2, axis=0)

    def attain_rotation_matrix_from_label_me_xml(self, xml_file_name):
        """
        Calculates the rotation matrix from a given LabelMe XML annotation file.

        :param xml_file_name: The XML file path.
        :type xml_file_name: str
        :return: The rotation matrix.
        :rtype: np.ndarray
        """
        line_segments = self.extract_line_segments_from_label_me_xml(xml_file_name=xml_file_name)
        line_segment_pairs = self.attain_line_segment_pairs_from_line_segments(line_segments=line_segments)
        vanishing_points = self.attain_vanishing_points_from_line_segment_pairs(line_segment_pairs=line_segment_pairs)
        for vanishing_point in vanishing_points:
            self.obtain_rotation_column(vanishing_point=vanishing_point)
        self.obtain_missing_rotation_column()
        return np.concatenate((self.r1, self.r2, self.r3), axis=1)

    def pitch(self):
        """
        Return the pitch from the current rotation matrix.

        :return: The pitch in radians.
        :rtype: float
        """
        base = atan2(self.r3[1], self.r3[2])
        # Force directions.
        if base > pi/2:
            return base - pi
        elif base < -pi/2:
            return base + pi
        else:
            return base

    def roll(self):
        """
        Return the roll from the current rotation matrix.

        :return: The roll in radians.
        :rtype: float
        """
        base = atan2(self.r2[0], self.r1[0])
        # Force directions.
        if base > pi/2:
            return base - pi
        elif base < -pi/2:
            return base + pi
        else:
            return base
