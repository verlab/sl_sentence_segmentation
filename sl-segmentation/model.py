# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sign language sequence tagging keras model."""

import tensorflow as tf

from args import FLAGS


def input_size():
  """Calculate the size of the input pose by desired components."""
  points = 0
  if 'pose_keypoints_2d' in FLAGS.input_components:
    points += 25
  if 'face_keypoints_2d' in FLAGS.input_components:
    points += 70
  if 'hand_left_keypoints_2d' in FLAGS.input_components:
    points += 21
  if 'hand_right_keypoints_2d' in FLAGS.input_components:
    points += 21

  if FLAGS.include_pose:
    points = points * 3
    
  return points


def get_model():
  """Create keras sequential model following the hyperparameters."""

  model_name = FLAGS.model_path.split('/')[-1].split('.')[0]

  model = tf.keras.Sequential(name=model_name)
  model.add(tf.keras.layers.InputLayer(input_shape=(None, input_size())))
  model.add(tf.keras.layers.Dropout(FLAGS.input_dropout))  # Random feature dropout

  # Add LSTM
  for _ in range(FLAGS.encoder_layers):
    rnn = tf.keras.layers.LSTM(FLAGS.hidden_size, return_sequences=True)
    if FLAGS.encoder_bidirectional:
      rnn = tf.keras.layers.Bidirectional(rnn)
    model.add(rnn)

  # Project and normalize to labels space
  #model.add(tf.keras.layers.Dense(2))
  #model.add(tf.keras.layers.Softmax())
  model.add(tf.keras.layers.Dense(1))
  model.add(tf.keras.layers.Activation(tf.keras.activations.sigmoid))

  return model

tf.keras.utils.get_custom_objects().clear()
@tf.keras.utils.register_keras_serializable(name='weighted_binary_crossentropy')
def weighted_binary_crossentropy(y_true, y_pred):
  """
  Implementation of weighted binary cross entropy loss.
  """
  weights=[1.0, FLAGS.class_weight]
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.convert_to_tensor(y_true)
  y_true = tf.cast(y_true, tf.float32)
  weights = tf.convert_to_tensor(weights, dtype=y_pred.dtype)

  epsilon = tf.constant(tf.keras.backend.epsilon(), y_pred.dtype.base_dtype)
  y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
  bce = weights[1]*y_true*tf.math.log(y_pred+epsilon) + weights[0]*(1-y_true)*tf.math.log(1-y_pred+epsilon)

  return -bce

def build_model():
  """Apply input shape, loss, optimizer, and metric to the model."""
  precision = tf.keras.metrics.Precision()
  recall = tf.keras.metrics.Recall()
  #iou = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])
  model = get_model()
  model.build()
  model.compile(
      #loss='binary_crossentropy',
      loss=weighted_binary_crossentropy,
      optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate),
      metrics=['accuracy', precision, recall],
      run_eagerly=True
  )
  model.summary()

  return model
