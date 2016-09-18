"""
Code for generating ground truth data to check the accelerometer data accuracy.
"""
from collections import namedtuple
from xml.etree import ElementTree

Point = namedtuple('Point', ['x', 'y'])
VanishingPoint = namedtuple('VanishingPoint', ['x', 'y', 'axis'])
LineSegment = namedtuple('Line', ['start', 'end', 'axis'])

class GroundTruth():
    """
    A class to generate ground truth to check the accelerometer data accuracy.
    """
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
            if len(points) != 2:
                continue
            point0 = Point(x=int(points[0][0].text), y=int(points[0][1].text))
            point1 = Point(x=int(points[1][0].text), y=int(points[1][1].text))
            axis = object_entry.find('attribute').text
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
            return a.x * b.y - a.y * b.x

        div = determinant(x_diff, y_diff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = Point(determinant(line_segment1.start, line_segment1.end),
                  determinant(line_segment2.start, line_segment2.end))
        x = determinant(d, x_diff) / div
        y = determinant(d, y_diff) / div
        return Point(x, y)
