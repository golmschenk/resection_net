"""
Code for dealing with reading and interacting with TFRecords outside of the main network.
"""

import tensorflow as tf


class GoTFRecordsReader:
    """
    A class for dealing with reading and interacting with TFRecords outside of the main network.
    """
    def __init__(self, file_name_queue):
        self.tfrecords_file_name = file_name_queue
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(self.tfrecords_file_name)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_height': tf.FixedLenFeature([], tf.int64),
                'image_width': tf.FixedLenFeature([], tf.int64),
                'image_depth': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label_height': tf.FixedLenFeature([], tf.int64),
                'label_width': tf.FixedLenFeature([], tf.int64),
                'label_depth': tf.FixedLenFeature([], tf.int64),
                'label_raw': tf.FixedLenFeature([], tf.string),
            })

        self.image_shape, self.label_shape = self.extract_shapes_from_tfrecords_features(features)

        flat_image = tf.decode_raw(features['image_raw'], tf.uint8)
        self.image = tf.reshape(flat_image, self.image_shape)

        flat_label = tf.decode_raw(features['label_raw'], tf.float32)
        self.label = tf.reshape(flat_label, self.label_shape)

    @staticmethod
    def extract_shapes_from_tfrecords_features(features):
        """
        Extracts the image and label shapes from the TFRecords' features. Uses a short TF session to do so.

        :param features: The recovered TFRecords' protobuf features.
        :type features: dict[str, tf.Tensor]
        :return: The image and label shape tuples.
        :rtype: (int, int, int), (int, int, int)
        """
        image_height_tensor = tf.cast(features['image_height'], tf.int64)
        image_width_tensor = tf.cast(features['image_width'], tf.int64)
        image_depth_tensor = tf.cast(features['image_depth'], tf.int64)
        label_height_tensor = tf.cast(features['label_height'], tf.int64)
        label_width_tensor = tf.cast(features['label_width'], tf.int64)
        label_depth_tensor = tf.cast(features['label_depth'], tf.int64)
        # To read the TFRecords file, we need to start a TF session (including queues to read the file name).
        with tf.Session() as session:
            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coordinator)
            image_height, image_width, image_depth, label_height, label_width, label_depth = session.run(
                [image_height_tensor, image_width_tensor, image_depth_tensor, label_height_tensor, label_width_tensor,
                 label_depth_tensor])
            coordinator.request_stop()
            coordinator.join(threads)
        image_shape = (image_height, image_width, image_depth)
        label_shape = (label_height, label_width, label_depth)
        return image_shape, label_shape

if __name__ == '__main__':
    tfrecords_inspector = GoTFRecordsReader('data/nyud_micro.tfrecords')
    print(self.image_shape)