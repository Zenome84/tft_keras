# Temporal Fusion Transformer in Readable TF2/Keras Format
A version of the [Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363) in TF2 that is lightweight, utilizes Keras layers, and ultimately readable and modifiable.

This version uses the [Functional Keras API](https://keras.io/guides/functional_api/) to allow for single input/output interfaces that support multi-inputs/outputs. This need arises from TFT having inputs/outputs of varied shapes, which as of today can only be implemented via the Function API. The result is to then call `model.fit([inputs...], [outputs...])` or `model.predict([inputs...])` all packaged neatly into the Keras framework.

The goal of this project is to make the TFT code both readable in its TF2 implementation and extendable/modifiable. The original implementation, found [here](https://github.com/google-research/google-research/tree/master/tft), along with other versions on the net did not conform to TF2/Keras style and used "excessive" code. This resulted in work that required extensive effort to proofread or tweak.

Additionally, the goal of this work is to stay as true to the original paper as possible, so as to serve as a baseline. The major modifications include:
* Reducing the excessive creation of multiple Dense layers with more coordinated reshaping and concatenation
* A slimmer `InterpretableMultiHeadSelfAttention` layer that utilizes Keras Attention layer
* Packaging both Linear/Categorical Embedding into a single `GenericEmbedding` layer

The code was tested on TF-2.11 and TF-2.15.

A demo on a small Kaggle dataset is provided in demo/

TODO:
* Implement wrappers for `fit` and `predict`
* Add tests to reproduce paper's results using `tf.DataSet`