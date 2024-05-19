# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An AlphaZero style model with a policy and value head."""

import collections
import functools
import os
from typing import Sequence
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
import keras.layers as layers
keras.config.disable_traceback_filtering()

def cascade(x, fns):
  for fn in fns:
    print ("Cascasing " + str(fn))
    x = fn(x)
  return x

conv_2d = functools.partial(layers.Conv2D, padding="same")


def batch_norm(name):
  return layers.BatchNormalization(name=name)



def residual_layer(inputs, num_filters, kernel_size, name):
  return cascade(inputs, [
      conv_2d(num_filters, kernel_size, name=f"{name}_res_conv1"),
      batch_norm(f"{name}_res_batch_norm1"),
      layers.Activation("relu"),
      conv_2d(num_filters, kernel_size, name=f"{name}_res_conv2"),
      batch_norm(f"{name}_res_batch_norm2"),
      lambda x: layers.add([x, inputs]),
      layers.Activation("relu"),
  ])


class TrainInput(collections.namedtuple(
    "TrainInput", "observation legals_mask policy value")):
  """Inputs for training the Model."""

  @staticmethod
  def stack(train_inputs):
    observation, legals_mask, policy, value = zip(*train_inputs)
    return TrainInput(
        np.array(observation, dtype=np.float32),
        np.array(legals_mask, dtype=bool),
        np.array(policy),
        np.expand_dims(value, 1))


class Losses(collections.namedtuple("Losses", "policy value l2")):
  """Losses from a training step."""

  @property
  def total(self):
    return self.policy + self.value + self.l2

  def __str__(self):
    return ("Losses(total: {:.3f}, policy: {:.3f}, value: {:.3f}, "
            "l2: {:.3f})").format(self.total, self.policy, self.value, self.l2)

  def __add__(self, other):
    return Losses(self.policy + other.policy,
                  self.value + other.value,
                  self.l2 + other.l2)

  def __truediv__(self, n):
    return Losses(self.policy / n, self.value / n, self.l2 / n)


class Model(keras.Model):
  """An AlphaZero style model with a policy and value head.

  This supports three types of models: mlp, conv2d and resnet.

  All models have a shared torso stack with two output heads: policy and value.
  They have same meaning as in the AlphaGo Zero and AlphaZero papers. The resnet
  model copies the one in that paper when set with width 256 and depth 20. The
  conv2d model is the same as the resnet except uses a conv+batchnorm+relu
  instead of the res blocks. The mlp model uses dense layers instead of conv,
  and drops batch norm.

  Links to relevant articles/papers:
    https://deepmind.com/blog/article/alphago-zero-starting-scratch has an open
      access link to the AlphaGo Zero nature paper.
    https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go
      has an open access link to the AlphaZero science paper.

  All are parameterized by their input (observation) shape and output size
  (number of actions), though the conv2d and resnet might only work with games
  that have spatial data (ie 3 non-batch dimensions, eg: connect four would
  work, but not poker).

  The depth is the number of blocks in the torso, where the definition of a
  block varies by model. For a resnet it's a resblock which is two conv2ds,
  batch norms and relus, and an addition. For conv2d it's a conv2d, a batch norm
  and a relu. For mlp it's a dense plus relu.

  The width is the number of filters for any conv2d and the number of hidden
  units for any dense layer.

  Note that this uses an explicit graph so that it can be used for inference
  and training from C++. It seems to also be 20%+ faster than using eager mode,
  at least for the unit test.
  """

  valid_model_types = ["mlp", "conv2d", "resnet"]

  def __init__(self, *args, **kwargs):
    """Init a model. Use build_model, from_checkpoint or from_graph instead."""
    super().__init__(*args, **kwargs)
    self.loss_tracker = keras.metrics.Mean(name="loss")

    def get_var(name):
      return self._session.graph.get_tensor_by_name(name + ":0")

#    self._input = get_var("input")
#    self._legals_mask = get_var("legals_mask")
#    self._training = get_var("training")
#    self._value_out = get_var("value_out")
#    self._policy_softmax = get_var("policy_softmax")
#    self._policy_loss = get_var("policy_loss")
#    self._value_loss = get_var("value_loss")
#    self._l2_reg_loss = get_var("l2_reg_loss")
#    self._policy_targets = get_var("policy_targets")
#    self._value_targets = get_var("value_targets")
#    self._train = self._session.graph.get_operation_by_name("train")

  @classmethod
  def build_model(cls, model_type, input_shape, output_size, nn_width, nn_depth,
                  weight_decay, learning_rate, path):
    """Build a model with the specified params."""
    if model_type not in cls.valid_model_types:
      raise ValueError(f"Invalid model type: {model_type}, "
                       f"expected one of: {cls.valid_model_types}")

    return cls._define_graph(model_type, input_shape, output_size, nn_width,
                        nn_depth, weight_decay, learning_rate)
  @classmethod
  def from_checkpoint(cls, checkpoint, path=None):
    """Load a model from a checkpoint."""
    model = cls.from_graph(checkpoint, path)
    model.load_checkpoint(checkpoint)
    return model

  @classmethod
  def from_graph(cls, metagraph, path=None):
    """Load only the model from a graph or checkpoint."""
    if not os.path.exists(metagraph):
      metagraph += ".meta"
    if not path:
      path = os.path.dirname(metagraph)
    g = tf.Graph()  # Allow multiple independent models and graphs.
    with g.as_default():
      saver = tf.train.import_meta_graph(metagraph)
    session = tf.Session(graph=g)
    session.__enter__()
    session.run("init_all_vars_op")
    return cls(session, saver, path)

  @classmethod
  def _define_graph(cls, model_type, input_shape, output_size,
                    nn_width, nn_depth, weight_decay, learning_rate):
    """Define the model graph."""
    # Inference inputs
    input_size = int(np.prod(input_shape))
    observations = keras.Input(dtype="float32", shape=(input_size,), name="input")
    legals_mask = keras.Input(dtype="bool", shape=(output_size,),
                                 name="legals_mask")

    # Main torso of the network
    if model_type == "mlp":
      torso = observations  # Ignore the input shape, treat it as a flat array.
      for i in range(nn_depth):
        torso = cascade(torso, [
            layers.Dense(nn_width, name=f"torso_{i}_dense"),
            layers.Activation("relu"),
        ])
    elif model_type == "conv2d":
      torso = layers.Reshape(input_shape)(observations)
      for i in range(nn_depth):
        torso = cascade(torso, [
            conv_2d(nn_width, 3, name=f"torso_{i}_conv"),
            batch_norm(f"torso_{i}_batch_norm"),
            layers.Activation("relu"),
        ])
    elif model_type == "resnet":
      torso = cascade(observations, [
          layers.Reshape(input_shape),
          conv_2d(nn_width, 3, name="torso_in_conv"),
          batch_norm("torso_in_batch_norm"),
          layers.Activation("relu"),
      ])
      for i in range(nn_depth):
        torso = residual_layer(torso, nn_width, 3, f"torso_{i}")
    else:
      raise ValueError("Unknown model type.")

    # The policy head
    if model_type == "mlp":
      policy_head = cascade(torso, [
          layers.Dense(nn_width, name="policy_dense"),
          layers.Activation("relu"),
      ])
    else:
      policy_head = cascade(torso, [
          conv_2d(filters=2, kernel_size=1, name="policy_conv"),
          batch_norm("policy_batch_norm"),
          layers.Activation("relu"),
          layers.Flatten(),
      ])
    print(f"policy_head.shape = {policy_head.shape}")
    policy_logits = layers.Dense(output_size, name="policy")(policy_head)
    print(f"policy_logits.shape = {policy_logits.shape}")
    print(f"output_size = {output_size}")

    class PolicyOutput(keras.Layer):
      def call(self, legals_mask, policy_logits):
        return layers.Softmax()(tf.where(legals_mask, policy_logits,
                             -1e32 * tf.ones_like(policy_logits)))

    policy_softmax = PolicyOutput(name="policy_softmax")(legals_mask, policy_logits)

    # The value head
    if model_type == "mlp":
      value_head = torso  # Nothing specific before the shared value head.
    else:
      value_head = cascade(torso, [
          conv_2d(filters=1, kernel_size=1, name="value_conv"),
          batch_norm("value_batch_norm"),
          layers.Activation("relu"),
          layers.Flatten(),
      ])
    value_out = cascade(value_head, [
        layers.Dense(nn_width, name="value_dense"),
        layers.Activation("relu"),
        layers.Dense(1, name="value"),
        layers.Activation("tanh"),
    ])
    # Need the identity to name the single value output from the dense layer.
    value_out = keras.layers.Identity(name="value_out")(value_out)

    model = cls(inputs=[observations, legals_mask], outputs=[policy_softmax, value_out], name=f"AlphaZero")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    return model

  def train_step(self, data):
    inputs, goal_outputs = data
    observations, legals_mask = inputs
    policy_targets, value_targets = goal_outputs

    with tf.GradientTape() as tape:
      pred_outputs = self(inputs, training=True)  # Forward pass
      policy_logits, value_target = pred_outputs
      policy_loss = keras.ops.mean(
        keras.ops.sparse_categorical_crossentropy(from_logits=True,
            output=policy_logits, target=policy_targets))
        
      value_loss = keras.ops.mean_squared_error(value_out, value_targets)

      l2_reg_loss = tf.add_n([
        weight_decay * tf.nn.l2_loss(var)
        for var in self.trainable_variables()
        if "/bias:" not in var.name
      ])

      total_loss = policy_loss + value_loss + l2_reg_loss

    trainable_vars = self.trainable_variables
    gradients = tape.gradient(total_loss, trainable_vars)

    # Update weights
    self.optimizer.apply(gradients, trainable_vars)

    # Compute our own metrics
    self.loss_tracker.update_state(total_loss)
    
    return {
      "loss": self.loss_tracker.result(),
    }

  @property
  def num_trainable_variables(self):
    return sum(np.prod(v.shape) for v in self.trainable_variables)

  def print_trainable_variables(self):
    for v in self.trainable_variables:
      print("{}: {}".format(v.name, v.shape))

  def write_graph(self, filename):
    full_path = os.path.join(self._path, filename)
    self.save(full_path)
    return full_path

  def inference(self, observation, legals_mask):
    return self.predict([observation, legals_mask])

  def save_checkpoint(self, path):
    print(f"About to save checkpoint to {path}")
    self.save_weights(path, overwrite=True)
    print(f"Saved checkpoint to {path}")

  def load_checkpoint(self, path):
    return self.load_weights(full_path, overwrite=True)
