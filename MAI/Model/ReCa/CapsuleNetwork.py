import tensorflow as tf
from tensorflow.python.keras import layers as tf_layers
from tensorflow.python.keras import Model
from MAI.Model.ReCa.CapsuleLayer import Capsule
from MAI.Model.ReCa.EmCapsuleLayer import EMCapsule
from MAI.Model.ReCa.GammaCapsuleLayer import GammaCapsule
from MAI.Model.ReCa.PrimaryCapsuleLayer import PrimaryCapsule
from MAI.Model.ReCa.ReconstructionLayer import ReconstructionLayer
from MAI.Model.ReCa.NormalizationLayer import Normalization
from MAI.Model.ReCa.ResidualLayer import ResLayer
from MAI.Utils.Params import DIMENSIONS, LAYERS, ROUTING, MAKE_SKIPS, NO_SKIPS, IMG_SIZE


class CapsNet(Model):
    def get_config(self):
        pass

    def __init__(self):
        super(CapsNet, self).__init__()

        dimensions = list(map(int, DIMENSIONS.split(",")))
        routing = ROUTING
        layers = list(map(int, LAYERS.split(",")))
        self.make_skips = MAKE_SKIPS
        self.skip_dist = NO_SKIPS

        # Create model
        CapsuleType = {
            "rba": Capsule,
            "em": EMCapsule,
            "sda": GammaCapsule
        }

        self.use_bias = True
        self.use_reconstruction = False
        self.num_classes = layers[-1]

        with tf.name_scope(self.name):
            self.reshape = tf_layers.Reshape(target_shape=[
                IMG_SIZE, IMG_SIZE, IMG_SIZE], input_shape=(IMG_SIZE, IMG_SIZE))

            channels = layers[0]
            dim = dimensions[0]
            self.conv_1 = tf_layers.Conv2D(
                channels * dim, (14, 14), kernel_initializer="he_normal", padding='valid', activation="relu")
            self.primary = PrimaryCapsule(
                name="PrimaryCapsuleLayer", channels=channels, dim=dim, kernel_size=(14, 14))
            self.capsule_layers = []

            size = 16 * 16 if (IMG_SIZE == 512) else \
                28 * 28 if (IMG_SIZE == 1024) else \
                    4 * 4
            for i in range(1, len(layers)):
                self.capsule_layers.append(
                    CapsuleType[routing](
                        name="CapsuleLayer%d" % i,
                        in_capsules=((size * channels) if i == 1 else layers[i - 1]),
                        in_dim=(dim if i == 1 else dimensions[i - 1]),
                        out_capsules=layers[i],
                        out_dim=dimensions[i],
                        use_bias=self.use_bias)
                )

            if self.use_reconstruction:
                self.reconstruction_network = ReconstructionLayer(
                    name="ReconstructionNetwork",
                    in_capsules=self.num_classes,
                    in_dim=dimensions[-1],
                    out_dim=IMG_SIZE,
                    img_dim=IMG_SIZE)
            self.norm = Normalization()
            self.residual = ResLayer()

    # Inference
    def call(self, x, y=None, mask=None):
        x = self.reshape(x)
        x = self.conv_1(x)
        x = self.primary(x)
        layers = [x]

        capsule_outputs = []
        for i, capsule in enumerate(self.capsule_layers):
            x = capsule(x)
            capsule_outputs.append(x)
            if i != len(self.capsule_layers) - 1:
                self.cb.append(x)

            # add skip connection
            if self.make_skips and i > 0 and i % self.skip_dist == 0:
                out_skip = capsule_outputs[i - self.skip_dist]
                if x.shape == out_skip.shape:
                    x = self.residual(x, out_skip)

            layers.append(x)
        r = self.reconstruction_network.call(
            x, y) if self.use_reconstruction else None
        out = self.norm.call(x)

        return out, r, layers
