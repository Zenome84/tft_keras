
import tensorflow as tf

layerDense = tf.keras.layers.Dense
layerEmbedding = tf.keras.layers.Embedding
layerReshape = tf.keras.layers.Reshape


class GenericEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_categories, embedding_size):
        super().__init__()

        if num_categories == 0:
            self._embedding = layerDense(embedding_size)
        else:
            self._embedding = layerEmbedding(num_categories, embedding_size)
        self._reshape = layerReshape([embedding_size])

    def call(self, inputs):
        x = self._embedding(inputs)
        x = self._reshape(x)
        return x
