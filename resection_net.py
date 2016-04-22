"""
Code related to the ResectionNet.
"""
import tensorflow as tf

from resection_data import ResectionData
from go_net import GoNet
from interface import Interface
from convenience import weight_variable, bias_variable, leaky_relu, conv2d


class ResectionNet(GoNet):
    """
    A neural network class to estimate camera parameters from 2D images.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = ResectionData()


if __name__ == '__main__':
    interface = Interface(network_class=ResectionNet)
    interface.train()
