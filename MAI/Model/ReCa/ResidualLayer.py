from tensorflow.python.keras import layers, models, Model


class ResLayer(layers.Layer):
    def call(self, out_previous, out_skip, **kwargs):
        x = layers.Add()([out_previous, out_skip])
        return x