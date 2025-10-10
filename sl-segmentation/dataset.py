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

"""Utilities to load and process a sign language detection dataset."""
import functools
import os
from typing import Any
from typing import Dict
from typing import Tuple

from pose_format.pose import Pose
from pose_format.pose_header import PoseHeader
from pose_format.tensorflow.masked.tensor import MaskedTensor
from pose_format.tensorflow.pose_body import TensorflowPoseBody
from pose_format.tensorflow.pose_body import TF_POSE_RECORD_DESCRIPTION
from pose_format.utils.reader import BufferReader
import tensorflow as tf

from args import FLAGS

import numpy as np


@functools.lru_cache(maxsize=1)
def get_openpose_header():
  """Get pose header with OpenPose components description."""
  dir_path = os.path.dirname(os.path.realpath(__file__))
  header_path = os.path.join(dir_path, "assets/openpose.poseheader")
  f = open(header_path, "rb")
  reader = BufferReader(f.read())
  header = PoseHeader.read(reader)
  return header


def differentiate_frames(src):
  """Subtract every two consecutive frames."""
  # Shift data to pre/post frames
  pre_src = src[:-1]
  post_src = src[1:]

  # Differentiate src points
  src = pre_src - post_src

  return src


def distance(src):
  """Calculate the Euclidean distance from x:y coordinates."""
  square = src.square()
  sum_squares = square.sum(axis=-1).fix_nan()
  sqrt = sum_squares.sqrt().zero_filled()
  return sqrt


def optical_flow(src, fps):
  """Calculate the optical flow norm between frames, normalized by fps."""

  # Remove "people" dimension
  src = src.squeeze(1)

  # Differentiate Frames
  src = differentiate_frames(src)

  # Calculate distance
  src = distance(src)

  # Normalize distance by fps
  src = src * fps

  return src

def flatten_pose(src):
  """Reshape the pose object to be a flatten array of the pose coordinates."""

  # compute array shape
  shape = [tf.shape(src)[k] for k in range(4)]

  # remove people dimension
  src = tf.reshape(src, [shape[0]*shape[1], shape[2], shape[3]])

  # compute array shape again
  shape = [tf.shape(src)[k] for k in range(3)]

  # reshape last two coordinates
  src = tf.reshape(src, [shape[0], shape[1]*2])

  # remove the first frame
  src = src[1:]

  return src

minimum_fps = tf.constant(1, dtype=tf.float32)


def load_datum(tfrecord_dict):
  """Convert tfrecord dictionary to tensors."""
  pose_body = TensorflowPoseBody.from_tfrecord(tfrecord_dict)
  pose = Pose(header=get_openpose_header(), body=pose_body)
  tgt = tf.io.decode_raw(tfrecord_dict["is_signing"], out_type=tf.int64)

  fps = pose.body.fps
  frames = tf.cast(tf.size(tgt), dtype=fps.dtype)

  # Get only relevant input components
  pose = pose.get_components(FLAGS.input_components)

  return {
      "fps": pose.body.fps,
      "frames": frames,
      "tgt": tgt,
      "pose_data_tensor": pose.body.data.tensor,
      "pose_data_mask": pose.body.data.mask,
      "pose_confidence": pose.body.confidence,
  }


def process_datum(datum,
                  augment=False):
  """Prepare every datum to be an input-output pair for training / eval.

  Supports data augmentation only including frames dropout.
  Frame dropout affects the FPS, which does change the optical flow.

  Args:
      datum (Dict[str, tf.Tensor]): a dictionary of tensors loaded from the
        tfrecord.
      augment (bool): should apply data augmentation on the datum?

  Returns:
     dict(Dict[str, tf.Tensor]): dictionary including "src" and "tgt" tensors
  """
  masked_tensor = MaskedTensor(
      tensor=datum["pose_data_tensor"], mask=datum["pose_data_mask"])
  pose_body = TensorflowPoseBody(
      fps=datum["fps"], data=masked_tensor, confidence=datum["pose_confidence"])
  pose = Pose(header=get_openpose_header(), body=pose_body)
  tgt = datum["tgt"]

  fps = pose.body.fps
  frames = datum["frames"]

  if augment:
    pose, selected_indexes = pose.frame_dropout_normal(dropout_std = FLAGS.frame_dropout_std)
    tgt = tf.gather(tgt, selected_indexes)

    new_frames = tf.cast(tf.size(tgt), dtype=fps.dtype)

    fps = tf.math.maximum(minimum_fps, (new_frames / frames) * fps)
    frames = new_frames

  flat = flatten_pose(datum['pose_data_tensor'])
  flow = optical_flow(pose.body.data, fps)
  tgt = tgt[1:]  # First frame tag is not used

  if FLAGS.include_pose:
    # concat both arrays
    src = tf.concat([flow, flat], axis = -1)
    return {"src": src, "tgt": tgt}
  else:
    return {"src": flow, "tgt": tgt}


def prepare_io(datum):
  """Convert dictionary into input-output tuple for Keras."""
  src = datum["src"]
  tgt = datum["tgt"]

  return src, tgt


def batch_dataset(dataset, batch_size):
  """Batch and pad a dataset."""
  dataset = dataset.padded_batch(
      batch_size, padded_shapes={
          "src": [None, None],
          "tgt": [None]
      })

  return dataset.map(prepare_io)


def train_pipeline(dataset):
  """Prepare the training dataset."""
  #dataset = dataset.map(load_datum).cache()
  dataset = dataset.map(load_datum)
  dataset = dataset.repeat()
  #dataset = dataset.map(lambda d: process_datum(d, True))
  dataset = dataset.map(process_datum) # sem augmentation
  dataset = dataset.shuffle(FLAGS.batch_size)
  dataset = batch_dataset(dataset, FLAGS.batch_size)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


def test_pipeline(dataset):
  """Prepare the test dataset."""
  dataset = dataset.map(load_datum)
  dataset = dataset.map(process_datum)
  dataset = batch_dataset(dataset, FLAGS.test_batch_size)
  #return dataset.cache()
  return dataset


def split_dataset(
    dataset
):
  """Split dataset to train, dev, and test."""

  # dev indexes
  dev_idx = np.loadtxt('./split/dev.csv')
  keys_dev = tf.constant(dev_idx, dtype=tf.int64)
  vals_dev = tf.ones_like(keys_dev)
  table_dev = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_dev, vals_dev),
    default_value=0)
  
  # test indexes
  test_idx = np.loadtxt('./split/test.csv')
  keys_test = tf.constant(test_idx, dtype=tf.int64)
  vals_test = tf.ones_like(keys_test)
  table_test = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_test, vals_test),
    default_value=0)
  
  # train indexes
  train_idx = np.loadtxt('./split/train.csv')
  keys_train = tf.constant(train_idx, dtype=tf.int64)
  vals_train = tf.ones_like(keys_train)
  table_train = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_train, vals_train),
    default_value=0)
  
  # check index
  def is_dev(x, _):
    table_value = table_dev.lookup(x)
    return tf.cast(table_value, tf.bool)

  def is_test(x, _):
    table_value = table_test.lookup(x)
    return tf.cast(table_value, tf.bool)

  def is_train(x, y):
    table_value = table_train.lookup(x)
    return tf.cast(table_value, tf.bool)

  def recover(_, y):
    return y

  # apply pipeline
  train = train_pipeline(dataset.enumerate().filter(is_train).map(recover))
  dev = test_pipeline(dataset.enumerate().filter(is_dev).map(recover))
  test = test_pipeline(dataset.enumerate().filter(is_test).map(recover))

  return train, dev, test


def get_datasets():
  """Get train, dev, and test datasets."""
  # Set features
  features = {"is_signing": tf.io.FixedLenFeature([], tf.string)}
  features.update(TF_POSE_RECORD_DESCRIPTION)

  # Dataset iterator
  dataset = tf.data.TFRecordDataset(filenames=[FLAGS.dataset_path])
  dataset = dataset.map(
      lambda serialized: tf.io.parse_single_example(serialized, features))

  return split_dataset(dataset)
