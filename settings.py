"""
Code for the settings of the network.
"""
from gonet.settings import Settings as GoSettings


class Settings(GoSettings):
    """
    A class for the settings of the network.
    """
    def __init__(self):
        super().__init__()

        self.network_name = 'resection_net'

        self.batch_size = 3
        self.initial_learning_rate = 0.0001
        self.learning_rate_decay_rate = 0.1
        self.learning_rate_decay_steps = 100000

        self.image_height = 464
        self.image_width = 624

        # The below settings are ResectionNet specific and should not be changed.
        self.label_height = 2
        self.label_width = 1

        self.data_directory = '/Volumes/ResectionNetHub/storage/data'
        self.datasets_json = '/Volumes/ResectionNetHub/storage/extra/nyudepth_datasets.json'
