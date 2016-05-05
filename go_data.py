"""
Code for managing the TFRecord data.
"""
import os
import h5py
import numpy as np
import tensorflow as tf


class GoData:
    """
    A class for managing the TFRecord data.
    """

    def __init__(self, data_directory='data', data_name='nyud', images_numpy_file_name='nyud_images',
                 labels_numpy_file_name='nyud_labels'):
        self.data_directory = data_directory
        self.data_name = data_name
        self.images_numpy_file_name = images_numpy_file_name
        self.labels_numpy_file_name = labels_numpy_file_name
        self.height = 464 // 8
        self.width = 624 // 8
        self.channels = 3
        self.original_height = 464
        self.original_width = 624
        self.images = None
        self.labels = None

    def read_and_decode(self, filename_queue):
        """
        A definition of how TF should read the file record.
        Slightly altered version from https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/how_tos/ \
                                      reading_data/fully_connected_reader.py

        :param filename_queue: The file name queue to be read.
        :type filename_queue: tf.QueueBase
        :return: The read file data including the image data and label data.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label_raw': tf.FixedLenFeature([], tf.string),
            })

        flat_image = tf.decode_raw(features['image_raw'], tf.uint8)
        unnormalized_image = tf.reshape(flat_image, [self.height, self.width, self.channels])
        image = tf.cast(unnormalized_image, tf.float32) * (1. / 255) - 0.5

        flat_label = tf.decode_raw(features['label_raw'], tf.float32)
        label = self.reshape_decoded_label(flat_label)

        return image, label

    def reshape_decoded_label(self, flat_label):
        """
        Reshapes the label decoded from the TF record. Allows easy overriding by sub classes.

        :param flat_label: The flat label from the decoded TF record.
        :type flat_label: tf.Tensor
        :return: The reshaped label.
        :rtype: tf.Tensor
        """
        return tf.reshape(flat_label, [self.height, self.width, 1])

    def inputs(self, data_type, batch_size, num_epochs=None):
        """
        Prepares the data inputs.
        Slightly altered version from https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/how_tos/ \
                                      reading_data/fully_connected_reader.py

        :param data_type: The type of data file (usually train, validation, or test).
        :type data_type: str
        :param batch_size: The size of the batches
        :type batch_size: int
        :param num_epochs: Number of epochs to run for. Infinite if None.
        :type num_epochs: int | None
        :return: The images and depths inputs.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        if data_type:
            file_name = self.data_name + '.' + data_type + '.tfrecords'
        else:
            file_name = self.data_name + '.tfrecords'
        file_path = os.path.join(self.data_directory, file_name)

        with tf.name_scope('Input'):
            filename_queue = tf.train.string_input_producer([file_path], num_epochs=num_epochs)

            image, label = self.read_and_decode(filename_queue)

            images, labels = tf.train.shuffle_batch(
                [image, label], batch_size=batch_size, num_threads=2,
                capacity=500 + 3 * batch_size, min_after_dequeue=500
            )

            return images, labels

    def convert_mat_file_to_numpy_file(self, mat_file_path, number_of_samples=None):
        """
        Generate image and depth numpy files from the passed mat file path.

        :param mat_file_path: The path to the mat file.
        :type mat_file_path: str
        :param number_of_samples: The number of samples to extract.
        :type number_of_samples: int
        """
        mat_data = h5py.File(mat_file_path, 'r')
        images = self.convert_mat_data_to_numpy_array(mat_data, 'images', number_of_samples=number_of_samples)
        images = self.crop_data(images)
        depths = self.convert_mat_data_to_numpy_array(mat_data, 'depths', number_of_samples=number_of_samples)
        depths = self.crop_data(depths)
        basename = os.path.basename(os.path.splitext(mat_file_path)[0])
        data_directory = os.path.dirname(mat_file_path)
        np.save(os.path.join(data_directory, 'images_' + basename) + '.npy', images)
        np.save(os.path.join(data_directory, 'depths_' + basename) + '.npy', depths)

    @staticmethod
    def convert_mat_data_to_numpy_array(mat_data, variable_name_in_mat_data, number_of_samples=None):
        """
        Converts a mat data variable to a numpy array.

        :param mat_data: The mat data containing the variable to be converted.
        :type mat_data: h5py.File
        :param variable_name_in_mat_data: The name of the variable to extract.
        :type variable_name_in_mat_data: str
        :param number_of_samples: The number of samples to extract.
        :type number_of_samples: int
        :return: The numpy array.
        :rtype: np.ndarray
        """
        mat_variable = mat_data.get(variable_name_in_mat_data)
        reversed_array = np.array(mat_variable)
        array = reversed_array.transpose()
        if variable_name_in_mat_data in ('images', 'depths'):
            array = np.rollaxis(array, -1)
        if number_of_samples:
            return array[:number_of_samples]
        else:
            return array

    @staticmethod
    def crop_data(array):
        """
        Crop the NYU data to remove dataless borders.

        :param array: The numpy array to crop
        :type array: np.ndarray
        :return: The cropped data.
        :rtype: np.ndarray
        """
        return array[:, 8:-8, 8:-8]

    def convert_mat_to_tfrecord(self, mat_file_path):
        """
        Converts the mat file data into a TFRecords file.

        :param mat_file_path: The path to mat file to convert.
        :type mat_file_path: str
        """
        mat_data = h5py.File(mat_file_path, 'r')
        uncropped_images = self.convert_mat_data_to_numpy_array(mat_data, 'images')
        self.images = self.crop_data(uncropped_images)
        uncropped_labels = self.convert_mat_data_to_numpy_array(mat_data, 'depths')
        self.labels = self.crop_data(uncropped_labels)
        self.shrink()
        self.convert_to_tfrecord()

    def numpy_files_to_tfrecords(self, augment=False):
        """
        Converts NumPy files to a TFRecords file.
        """
        self.load_numpy_files()
        self.shrink()
        if augment:
            self.augment_data_set()
        self.convert_to_tfrecord()

    def load_numpy_files(self):
        """
        Loads data from the numpy files into the object.
        """
        images_numpy_file_path = os.path.join(self.data_directory, self.images_numpy_file_name)
        labels_numpy_file_path = os.path.join(self.data_directory, self.labels_numpy_file_name)
        self.images = np.load(images_numpy_file_path)
        self.labels = np.load(labels_numpy_file_path)

    def convert_to_tfrecord(self):
        """
        Converts the data to a TFRecord.
        """
        number_of_examples = self.labels.shape[0]
        if self.images.shape[0] != number_of_examples:
            raise ValueError("Images count %d does not match label count %d." %
                             (self.images.shape[0], number_of_examples))
        rows = self.images.shape[1]
        cols = self.images.shape[2]
        depth = self.images.shape[3]

        filename = os.path.join(self.data_directory, self.data_name + '.tfrecords')
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(number_of_examples):
            image_raw = self.images[index].tostring()
            label_raw = self.labels[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'channels': _int64_feature(depth),
                'image_raw': _bytes_feature(image_raw),
                'label_raw': _bytes_feature(label_raw),
            }))
            writer.write(example.SerializeToString())

    def shrink(self):
        """
        Rebins the data arrays into the specified data size.
        """
        self.images = self.shrink_array_with_rebinning(self.images)
        self.labels = self.shrink_array_with_rebinning(self.labels)

    def shrink_array_with_rebinning(self, array):
        """
        Rebins the NumPy array into a new size, averaging the bins between.
        :param array: The array to resize.
        :type array: np.ndarray
        :return: The resized array.
        :rtype: np.ndarray
        """
        compression_shape = [
            array.shape[0],
            self.height,
            array.shape[1] // self.height,
            self.width,
            array.shape[2] // self.width,
        ]
        if len(array.shape) == 4:
            compression_shape.append(self.channels)
            return array.reshape(compression_shape).mean(4).mean(2).astype(np.uint8)
        else:
            return array.reshape(compression_shape).mean(4).mean(2)

    def gaussian_noise_augmentation(self, standard_deviation, number_of_variations):
        """
        Applies random gaussian noise to the images.

        :param standard_deviation: The standard deviation of the gaussian noise.
        :type standard_deviation: float
        :param number_of_variations: The number of noisy copies to create.
        :type number_of_variations: int
        """
        augmented_images_list = [self.images]
        augmented_labels_list = [self.labels]
        for _ in range(number_of_variations):
            # noinspection PyTypeChecker
            augmented_images_list.append(np.random.normal(self.images.astype(np.int16),
                                                          standard_deviation).clip(0, 255).astype(np.uint8))
            augmented_labels_list.append(self.labels)
        self.images = np.concatenate(augmented_images_list)
        self.labels = np.concatenate(augmented_labels_list)

    @staticmethod
    def offset_array(array, offset, axis):
        """
        Offsets an array by the given amount (simply by copying the array to the given portion).
        Note, this is only working for very specific cases at the moment.

        :param array: The array to offset.
        :type array: np.ndarray
        :param offset: The amount of the offset.
        :type offset: int
        :param axis: The axis to preform the offset on.
        :type axis: int
        :return: The offset array.
        :rtype: np.ndarray
        """
        offset_array = np.copy(array)
        offset_array = np.swapaxes(offset_array, 0, axis)
        if offset > 0:
            offset_array[offset:] = offset_array[:-offset]
        else:
            offset_array[:offset] = offset_array[-offset:]
        offset_array = np.swapaxes(offset_array, 0, axis)
        return offset_array

    def offset_augmentation(self, offset_limit):
        """
        Augments the data using a crude spatial shifting based on a given offset.

        :param offset_limit: The value of the maximum offset.
        :type offset_limit: int
        """
        augmented_images_list = [self.images]
        augmented_labels_list = [self.labels]
        for axis in [1, 2]:
            for offset in range(-offset_limit, offset_limit + 1):
                if offset == 0:
                    continue
                augmented_images_list.append(self.offset_array(self.images, offset, axis))
                augmented_labels_list.append(self.offset_array(self.labels, offset, axis))
        self.images = np.concatenate(augmented_images_list)
        self.labels = np.concatenate(augmented_labels_list)

    def augment_data_set(self):
        """
        Augments the data set with some basic approaches

        :param images: The images array.
        :type images: np.ndarray
        :param labels: The labels array.
        :type labels: np.ndarray
        :return: The images and the labels
        :rtype: (np.ndarray, np.ndarray)
        """
        print('Augmenting with spatial jittering...')
        self.offset_augmentation(1)
        print('Augmenting with gaussian noise...')
        self.gaussian_noise_augmentation(10, 4)

    def shuffle(self):
        """
        Shuffles the images and labels together.
        """
        permuted_indexes = np.random.permutation(len(self.images))
        self.images = self.images[permuted_indexes]
        self.labels = self.labels[permuted_indexes]


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    os.nice(10)

    data = GoData()
    data.convert_mat_to_tfrecord('data/nyu_depth_v2_labeled.mat')
