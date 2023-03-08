
import tensorflow as tf

layerDense = tf.keras.layers.Dense
layerDropout = tf.keras.layers.Dropout
layerReshape = tf.keras.layers.Reshape
layerSoftmax = tf.keras.layers.Softmax
layerLayerNormalization = tf.keras.layers.LayerNormalization


class GatedLinearUnit(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self._linear = layerDense(units)
        self._sigmoid = layerDense(units, activation="sigmoid")

    def call(self, inputs):
        return self._linear(inputs) * self._sigmoid(inputs)


def drop_gate_skip_norm(gate, residual, dropout_rate):
    x = layerDropout(dropout_rate)(gate)
    x = residual + GatedLinearUnit(residual.shape[-1])(x)
    x = layerLayerNormalization()(x)
    return x


class GatedResidualNetwork(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate, output_size: int=None):
        super().__init__()
        # self.units = units
        self._elu_dense = layerDense(units, activation="elu")
        self._linear_dense = layerDense(units)
        self._dropout = layerDropout(dropout_rate)
        self._layer_norm = layerLayerNormalization()
        if output_size is None:
            self._project = None
            self._gated_linear_unit = GatedLinearUnit(units)
        else:
            self._project = layerDense(output_size)
            self._gated_linear_unit = GatedLinearUnit(output_size) # TODO: check if None or != units

    def call(self, inputs):
        if type(inputs) == list:
            a = tf.keras.layers.concatenate(inputs[:-1])
            x = tf.keras.layers.concatenate(inputs)
        else:
            a = inputs
            x = inputs
        x = self._elu_dense(x)
        x = self._linear_dense(x)
        x = self._dropout(x)
        if self._project is not None:
            a = self._project(a)
        x = a + self._gated_linear_unit(x)
        x = self._layer_norm(x)
        return x


class VariableSelectionNetwork(tf.keras.layers.Layer):
    def __init__(self, num_features, units, dropout_rate, with_context=False):
        super().__init__()
        self._with_context = with_context
        # Create a GRN for each feature independently
        self._grns = [
            GatedResidualNetwork(units, dropout_rate)
            for _ in range(num_features)
        ]
        # Create a GRN for the concatenation of all the features
        self._grn_concat = GatedResidualNetwork(units, dropout_rate, num_features)
        self._softmax = layerSoftmax()

    def call(self, inputs):
        if self._with_context:
            v = inputs
        else:
            v = tf.keras.layers.concatenate(inputs)
        v = self._grn_concat(v)
        v = tf.expand_dims(self._softmax(v), axis=-1)

        x = []
        for idx, input in enumerate(inputs):
            if self._with_context and idx == len(self._grns):
                break
            x.append(self._grns[idx](input))
        x = tf.stack(x, axis=-2)

        outputs = tf.squeeze(tf.matmul(v, x, transpose_a=True), axis=-2)
        return outputs, x
