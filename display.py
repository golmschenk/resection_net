"""
Code for displaying results of programs and scripts.
"""
import math
import numpy as np
import os

from gonet.tfrecords_processor import TFRecordsProcessor
from resection_data import ResectionData


class Display:
    """
    A class for displaying results of programs and scripts.
    """
    @staticmethod
    def print_size_of_dataset(data_type):
        """
        Gives the number of images in a dataset.

        :param data_type: The data type being considered from the datasets.
        :type data_type: str
        """
        go_tfrecords_processor = TFRecordsProcessor()
        data = ResectionData()
        file_names = data.file_names_from_json(data_type)
        total_size = 0
        for file_name in file_names:
            print('Processing {}...'.format(file_name))
            _, labels = go_tfrecords_processor.read_to_numpy(os.path.join(data.settings.data_directory, file_name))
            total_size += labels.shape[0]
        print('The {} dataset has {} examples.'.format(data_type, total_size))

    @staticmethod
    def compare_dataset_labels(dataset1_path, dataset2_path):
        """
        Compares the labels of two datasets and displays various statistics.

        :param dataset1_path: The path to the first dataset.
        :type dataset1_path: str
        :param dataset2_path: The path to the second dataset.
        :type dataset2_path: str
        """
        go_tfrecords_processor = TFRecordsProcessor()
        _, labels1 = go_tfrecords_processor.read_to_numpy(dataset1_path)
        _, labels2 = go_tfrecords_processor.read_to_numpy(dataset2_path)
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
        
    @staticmethod
    def test_run_statistics(predicted_test_labels, test_labels):
        """
        Displays the accuracy statistics of a test run.

        :param predicted_test_labels: The predicted labels.
        :type predicted_test_labels: np.ndarray
        :param test_labels: The true labels.
        :type test_labels:  np.ndarray
        """
        absolute_difference = np.abs(predicted_test_labels - test_labels)
        squared_difference = np.square(absolute_difference)
        axis_mean_labels = np.mean(test_labels, axis=0)
        absolute_deviation = np.abs(test_labels - axis_mean_labels)
        mean_difference = np.mean(absolute_difference)
        combined_mean_squared_difference = np.mean(squared_difference)
        axis_mean_difference = np.mean(absolute_difference, axis=0)
        axis_mean_squared_difference = np.mean(squared_difference, axis=0)
        axis_labels_standard_deviation = np.std(test_labels, axis=0)
        axis_mean_absolute_deviation = np.mean(absolute_deviation, axis=0)

        print('Radians')
        print('Combined mean absolute difference: {}'.format(mean_difference))
        print('Combined mean squared difference: {}'.format(combined_mean_squared_difference))
        print('Pitch mean absolute difference: {}'.format(axis_mean_difference[0]))
        print('Pitch mean squared difference: {}'.format(axis_mean_squared_difference[0]))
        print('Roll mean absolute difference: {}'.format(axis_mean_difference[1]))
        print('Roll mean squared difference: {}'.format(axis_mean_squared_difference[1]))
        print('Ground truth pitch mean: {}'.format(axis_mean_labels[0]))
        print('Ground truth roll mean: {}'.format(axis_mean_labels[1]))
        print('Ground truth combined standard deviation: {}'.format(np.std(test_labels)))
        print('Ground truth pitch standard deviation: {}'.format(axis_labels_standard_deviation[0]))
        print('Ground truth roll standard deviation: {}'.format(axis_labels_standard_deviation[1]))
        print('Ground truth combined mean absolute deviation: {}'.format(np.mean(absolute_deviation)))
        print('Ground truth pitch mean absolute deviation: {}'.format(axis_mean_absolute_deviation[0]))
        print('Ground truth roll mean absolute deviation: {}'.format(axis_mean_absolute_deviation[1]))

        predicted_test_labels = np.degrees(predicted_test_labels)
        test_labels = np.degrees(test_labels)
        absolute_difference = np.abs(predicted_test_labels - test_labels)
        squared_difference = np.square(absolute_difference)
        axis_mean_labels = np.mean(test_labels, axis=0)
        absolute_deviation = np.abs(test_labels - axis_mean_labels)
        mean_difference = np.mean(absolute_difference)
        combined_mean_squared_difference = np.mean(squared_difference)
        axis_mean_difference = np.mean(absolute_difference, axis=0)
        axis_mean_squared_difference = np.mean(squared_difference, axis=0)
        axis_labels_standard_deviation = np.std(test_labels, axis=0)
        axis_mean_absolute_deviation = np.mean(absolute_deviation, axis=0)

        print('Degrees')
        print('Combined RMSE: {}'.format(np.sqrt(combined_mean_squared_difference)))
        print('Pitch RMSE: {}'.format(np.sqrt(axis_mean_squared_difference[0])))
        print('Roll RMSE: {}'.format(np.sqrt(axis_mean_squared_difference[1])))
        print('Ground truth pitch mean: {}'.format(axis_mean_labels[0]))
        print('Ground truth roll mean: {}'.format(axis_mean_labels[1]))
        print('Ground truth combined standard deviation: {}'.format(np.mean(np.std(test_labels, axis=0))))
        print('Combined mean squared difference: {}'.format(combined_mean_squared_difference))
        print('Ground truth pitch standard deviation: {}'.format(axis_labels_standard_deviation[0]))
        print('Pitch mean squared difference: {}'.format(axis_mean_squared_difference[0]))
        print('Ground truth roll standard deviation: {}'.format(axis_labels_standard_deviation[1]))
        print('Roll mean squared difference: {}'.format(axis_mean_squared_difference[1]))
        print('Ground truth combined mean absolute deviation: {}'.format(np.mean(absolute_deviation)))
        print('Combined mean absolute difference: {}'.format(mean_difference))
        print('Ground truth pitch mean absolute deviation: {}'.format(axis_mean_absolute_deviation[0]))
        print('Pitch mean absolute difference: {}'.format(axis_mean_difference[0]))
        print('Ground truth roll mean absolute deviation: {}'.format(axis_mean_absolute_deviation[1]))
        print('Roll mean absolute difference: {}'.format(axis_mean_difference[1]))


if __name__ == '__main__':
    display = Display()
    display.compare_dataset_labels('nyu_depth_v2_labeled.tfrecords', 'sunrgbd_kv1_NYUdata.tfrecords')
