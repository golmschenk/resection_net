"""
Code for generating ground truth data to check the accelerometer data accuracy.
"""
from collections import namedtuple
from xml.etree import ElementTree

Point = namedtuple('Point', ['x', 'y'])
LineSegment = namedtuple('Line', ['start', 'end', 'type'])

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
            type_ = object_entry.find('name').text
            line_segment = LineSegment(start=point0, end=point1, type=type_)
            line_segments.append(line_segment)
        return line_segments
