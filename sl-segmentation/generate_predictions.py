import tensorflow as tf
from tensorflow.keras.models import load_model

from args import FLAGS
from dataset import get_datasets

import numpy as np
import os
import sys

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

FLAGS(sys.argv)

# get dataset
train, dev, test = get_datasets()

# set model name
model_name = FLAGS.model_path.split('/')[-1].split('.')[0]

# load model
best_model = load_model(FLAGS.model_path, compile=False)
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
best_model.compile(
      loss=weighted_binary_crossentropy,
      optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate),
      metrics=['accuracy', precision, recall],
      run_eagerly=True
  )

# get metrics results
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