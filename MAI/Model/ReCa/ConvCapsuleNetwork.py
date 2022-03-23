import tensorflow as tf
import math
from tensorflow.keras import layers, models, Model
from tensorflow.keras.layers import BatchNormalization

from CapsuleLayer import Capsule
from EmCapsuleLayer import EMCapsule
from GammaCapsuleLayer import GammaCapsule
from ConvCapsLayer import ConvCapsule
from PrimaryCapsuleLayer import PrimaryCapsule
from ReconstructionLayer import ReconstructionLayer
from NormalizationLayer import Normalization
from ResidualLayer import ResLayer


class ConvCapsNet(Model):
    def __init__(self, args):
        super(ConvCapsNet, self).__init__()

        dimensions = list(map(int, args.dimension.split(","))) if args.dimension != "" else []
        layers = list(map(int, args.layers.split(","))) if args.layers != "" else []

        self.use_bias = args.use_bias
        self.use_reconstruction = args.use_reconstruction
        self.make_skips = args.make_skips
        self.skip_dist = args.skip_dist

        CapsuleType = {
            "rba": Capsule,
            "em": EMCapsule,
            "sda": GammaCapsule
        }

        img_size = 526

        conv1_filters, conv1_kernel, conv1_stride = 128, 7, 2
