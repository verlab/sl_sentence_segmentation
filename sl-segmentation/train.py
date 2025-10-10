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

"""Training script for sign language detection."""

import random

from absl import app
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import History
from tensorflow.keras.models import load_model

from args import FLAGS
from dataset import get_datasets
from model import build_model

import matplotlib.pyplot as plt
import numpy as np
import gc

import os
import time

class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()

def set_seed():
  """Set seed for deterministic random number generation."""
  seed = FLAGS.seed if FLAGS.seed is not None else random.randint(0, 1000)
  tf.random.set_seed(seed)
  random.seed(seed)
  
def safe_div(a, b):
  try:
    return a/b
  except ZeroDivisionError:
    return 0

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

def main(unused_argv):
  """Keras training loop with early-stopping and model checkpoint."""

  set_seed()

  tf.config.run_functions_eagerly(True)

  # create model folder if not exists
  model_folder = os.path.dirname(FLAGS.model_path)
  if not os.path.exists(model_folder):
     os.makedirs(model_folder)

  # set model name
  model_name = FLAGS.model_path.split('/')[-1].split('.')[0]

  # Initialize Dataset
  train, dev, test = get_datasets()
  
  # Initialize Model
  model = build_model()

  # Train
  es = EarlyStopping(
      monitor='val_loss',
      mode='min',
      verbose=1,
      patience=FLAGS.stop_patience)
  mc = ModelCheckpoint(
      FLAGS.model_path,
      monitor='val_loss',
      mode='min',
      verbose=1,
      save_best_only=True)
  hs = History()
  cl = ClearMemory()

  print('\nTraining:')
  start_time = time.time()
  with tf.device(FLAGS.device):
    model.fit(
        train,
        epochs=FLAGS.epochs,
        steps_per_epoch=FLAGS.steps_per_epoch,
        validation_data=dev,
        callbacks=[es, mc, hs, cl])
    
  end_time = time.time()
  training_time = round(end_time-start_time, 4)
  print(f'\nTraining time (in seconds): {training_time}')

  # # save history
  # max_accuracy = hs.history['val_accuracy'].index(max(hs.history['val_accuracy']))

  # plt.plot(hs.history['accuracy'])
  # plt.plot(hs.history['val_accuracy'])
  # plt.axvline(x=max_accuracy, linestyle='--', color='gray')
  # plt.title('Model')
  # plt.ylabel('Accuracy')
  # plt.xlabel('Epoch')
  # plt.legend(['train','val'], loc='lower right')
  # plt.savefig(f'./results_prov/plots/{model_name}_acc.png')

  plot_folder = './results/plots/'
  if not os.path.exists(plot_folder):
     os.makedirs(plot_folder)

  min_loss = hs.history['val_loss'].index(min(hs.history['val_loss']))

  plt.plot(hs.history['loss'])
  plt.plot(hs.history['val_loss'])
  plt.axvline(x=min_loss, linestyle='--', color='gray')
  plt.title('Model')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['train','val'], loc='upper right')
  plt.savefig(os.path.join(plot_folder,f'{model_name}_loss.png'))

  # save loss and accuracy results
  print('\nEvaluation:')
  print('\n - Train eval:')
  result_train = {'loss': hs.history['loss'][min_loss], 'accuracy': hs.history['accuracy'][min_loss],
                  'precision': hs.history['precision'][min_loss], 'recall': hs.history['recall'][min_loss],
                  #'iou': hs.history['io_u'][min_loss],
                  'f1': safe_div(2*hs.history['precision'][min_loss]*hs.history['recall'][min_loss], hs.history['precision'][min_loss]+hs.history['recall'][min_loss])}
  print(result_train)

  print('\n - Dev eval:')
  result_dev = {'loss': hs.history['val_loss'][min_loss], 'accuracy': hs.history['val_accuracy'][min_loss],
                'precision': hs.history['val_precision'][min_loss], 'recall': hs.history['val_recall'][min_loss],
                #'iou': hs.history['val_io_u'][min_loss],
                'f1': safe_div(2*hs.history['val_precision'][min_loss]*hs.history['val_recall'][min_loss], hs.history['val_precision'][min_loss]+hs.history['val_recall'][min_loss])}
  print(result_dev)

  print('\n - Test eval:')
  best_model = load_model(FLAGS.model_path, compile=False)
  precision = tf.keras.metrics.Precision()
  recall = tf.keras.metrics.Recall()
  best_model.compile(
      loss=weighted_binary_crossentropy,
      optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate),
      metrics=['accuracy', precision, recall],
      run_eagerly=True
  )
  result_test = best_model.evaluate(test, return_dict=True, verbose=0)
  result_test['f1'] = safe_div(2*result_test['precision']*result_test['recall'],result_test['precision']+result_test['recall'])
  print(result_test)

  # save predictions on dev and test set
  predictions_folder = './results/predictions/'
  if not os.path.exists(predictions_folder):
     os.makedirs(predictions_folder)
  # dev
  predictions = best_model.predict(dev, verbose=0).numpy()
  np.save(os.path.join(predictions_folder, f'{model_name}_dev'), predictions)
  # test
  predictions = best_model.predict(test, verbose=0).numpy()
  np.save(os.path.join(predictions_folder, f'{model_name}_test'), predictions)

if __name__ == '__main__':
  app.run(main)
