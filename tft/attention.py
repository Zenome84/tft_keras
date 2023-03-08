
import tensorflow as tf

layerDense = tf.keras.layers.Dense
layerReshape = tf.keras.layers.Reshape
layerSoftmax = tf.keras.layers.Softmax
layerAttention = tf.keras.layers.Attention
layerTimeDistributed = tf.keras.layers.TimeDistributed


class InterpretableMultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, heads, units, dropout_rate):
        super().__init__()
        d_att = units // heads
        self._queries = layerDense(heads*d_att, use_bias=False)
        self._keys = layerDense(heads*d_att, use_bias=False)
        self._values = layerDense(d_att, use_bias=False)

        self._reshape = layerTimeDistributed(layerReshape([heads, d_att]))
        self._softmax = layerSoftmax()
        
        self._attention = layerAttention(dropout=dropout_rate)
        self._out_weights = layerDense(units, use_bias=False)

    def call(self, inputs):
        query, value, key = inputs, inputs, inputs

        query = self._reshape(self._queries(query))
        key = self._reshape(self._keys(key))

        value = tf.repeat(tf.expand_dims(self._values(value), -2), tf.shape(key)[-2], -2)

        outputs, attention = self._attention(
            [query/tf.sqrt(tf.cast(tf.shape(key)[-1], float)), value, key],
            use_causal_mask=True, return_attention_scores=True
        )

        outputs = self._out_weights(tf.reduce_mean(outputs, -2))
        attention = tf.reduce_mean(attention, -2)

        return outputs, attention
