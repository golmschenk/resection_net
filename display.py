"""
Code for displaying results of programs and scripts.
"""
import numpy as np
from gonet.tfrecords_reader import TFRecordsReader


class Display:
    """
    A class for displaying results of programs and scripts.
    """
    @staticmethod
    def compare_dataset_labels(dataset1_path, dataset2_path):
        """
        Compares the labels of two datasets and displays various statistics.

        :param dataset1_path: The path to the first dataset.
        :type dataset1_path: str
        :param dataset2_path: The path to the second dataset.
        :type dataset2_path: str
        """
        go_tfrecords_reader = TFRecordsReader()
        _, labels1 = go_tfrecords_reader.convert_to_numpy(dataset1_path)
        _, labels2 = go_tfrecords_reader.convert_to_numpy(dataset2_path)
        absolute_difference = np.abs(labels1 - labels2)
        combined_mean_difference = np.mean(absolute_difference)
        combined_standard_deviation_difference = np.std(absolute_difference)
        column_mean_difference = np.mean(absolute_difference, axis=0)
        column_standard_deviation_difference = np.std(absolute_difference, axis=0)
        pitch_mean_difference = column_mean_difference[0]
        roll_mean_difference = column_mean_difference[1]
        pitch_standard_deviation_difference = column_standard_deviation_difference[0]
        roll_standard_deviation_difference = column_standard_deviation_difference[1]
        squared_difference = np.square(absolute_difference)
        combined_mean_squared_difference = np.mean(squared_difference)
        combined_standard_deviation_squared_difference = np.std(squared_difference)
        column_mean_squared_difference = np.mean(squared_difference, axis=0)
        column_standard_deviation_squared_difference = np.std(squared_difference, axis=0)
        pitch_mean_squared_difference = column_mean_squared_difference[0]
        roll_mean_squared_difference = column_mean_squared_difference[1]
        pitch_standard_deviation_squared_difference = column_standard_deviation_squared_difference[0]
        roll_standard_deviation_squared_difference = column_standard_deviation_squared_difference[1]
        print('combined_mean_difference', combined_mean_difference)
        print('combined_standard_deviation_difference', combined_standard_deviation_difference)
        print('pitch_mean_difference', pitch_mean_difference)
        print('roll_mean_difference', roll_mean_difference)
        print('pitch_standard_deviation_difference', pitch_standard_deviation_difference)
        print('roll_standard_deviation_difference', roll_standard_deviation_difference)
        print('combined_mean_squared_difference', combined_mean_squared_difference)
        print('combined_standard_deviation_squared_difference', combined_standard_deviation_squared_difference)
        print('pitch_mean_squared_difference', pitch_mean_squared_difference)
        print('roll_mean_squared_difference', roll_mean_squared_difference)
        print('pitch_standard_deviation_squared_difference', pitch_standard_deviation_squared_difference)
        print('roll_standard_deviation_squared_difference', roll_standard_deviation_squared_difference)


if __name__ == '__main__':
    display = Display()
    display.compare_dataset_labels('/Users/golmschenk/Code/resectionnet/data/nyudepth_micro.tfrecords', '/Users/golmschenk/Code/resectionnet/data/nyudepth_micro.tfrecords')