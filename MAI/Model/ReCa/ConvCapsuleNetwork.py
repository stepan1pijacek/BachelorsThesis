import tensorflow as tf
import math
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Reshape

from CapsuleLayer import Capsule
from EmCapsuleLayer import EMCapsule
from GammaCapsuleLayer import GammaCapsule
from ConvCapsLayer import ConvCapsule
from PrimaryCapsuleLayer import PrimaryCapsule
from ReconstructionLayer import ReconstructionLayer
from NormalizationLayer import Normalization
from ResidualLayer import ResLayer

from MAI.Utils.Params import IMG_SIZE, LAYERS, DIMENSIONS, NO_SKIPS, MAKE_SKIPS, ITERATIONS


class ConvCapsNet(Model):
    def __init__(self):
        super(ConvCapsNet, self).__init__()

        img_size = IMG_SIZE
        dimensions = list(map(int, DIMENSIONS.split(",")))
        layers = list(map(int, LAYERS.split(",")))

        self.use_bias = True
        self.use_reconstruction = True
        self.make_skips = MAKE_SKIPS
        self.skip_dist = NO_SKIPS

        conv1_filters, conv1_kernel, conv1_stride = 256, 7, 2
        out_width, out_height = (img_size - conv1_kernel) // conv1_stride + 1

        with tf.name_scope(self.name):

            # normal convolution
            self.conv_1 = Conv2D(
                conv1_filters,
                kernel_size=conv1_kernel,
                strides=conv1_stride,
                padding='valid',
                activation="relu",
                name="conv1"
            )

            # reshape into capsule shape
            self.capsuleShape = Reshape(
                target_shape=(
                    out_height, out_width, 1, conv1_filters
                ),
                name='toCapsuleShape'
            )

            self.capsule_layers = []
            for i in range(len(layers) - 1):
                self.capsule_layers.append(
                    ConvCapsule(
                        name="ConvCapsuleLayer" + str(i),
                        in_capsules=layers[i],
                        in_dim=dimensions[i],
                        out_dim=dimensions[i],
                        out_capsules=layers[i + 1],
                        kernel_size=3,
                        routing_iterations=ITERATIONS,
                        routing="rba"))

            # flatten for input to FC capsule
            self.flatten = tf.keras.layers.Reshape(target_shape=(out_height * out_width * layers[-2], dimensions[-2]),
                                                   name='flatten')

            # fully connected caspule layer
            self.fcCapsuleLayer = Capsule(
                name="FCCapsuleLayer",
                in_capsules=out_height * out_width * layers[-2],
                in_dim=dimensions[-2],
                out_capsules=layers[-1],
                out_dim=dimensions[-1],
                use_bias=self.use_bias)

            if self.use_reconstruction:
                self.reconstruction_network = ReconstructionLayer(
                    name="ReconstructionNetwork",
                    in_capsules=layers[-1],
                    in_dim=dimensions[-1],
                    out_dim=IMG_SIZE,
                    img_dim=IMG_SIZE)

            self.norm = Normalization()
            self.residual = ResLayer()

        # Inference
        def call(self, x, y):
            x = self.conv_1(x)
            x = self.capsuleShape(x)

            layers = []
            capsule_outputs = []
            i = 0
            for j, capsuleLayer in enumerate(self.capsule_layers):
                x = capsuleLayer(x)

                # add skip connection
                capsule_outputs.append(x)
                if self.make_skips and i > 0 and i % self.skip_dist == 0:
                    out_skip = capsule_outputs[j - self.skip_dist]
                    if x.shape == out_skip.shape:
                        # print('make residual connection from ', j-self.skip_dist, ' to ', j)
                        x = self.residual(x, out_skip)
                        i = -1

                i += 1
                layers.append(x)

            x = self.flatten(x)
            x = self.fcCapsuleLayer(x)

            r = self.reconstruction_network(x, y) if self.use_reconstruction else None
            out = self.norm(x)

            return out, r, layers
