import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from args import FLAGS
from dataset import get_datasets
import numpy as np
import sys

FLAGS(sys.argv)

def safe_div(a, b):
  try:
    return a/b
  except ZeroDivisionError:
    return 0

# define prediction based on probabilities
def get_prediction(prob):    
    pred = []
    for entry in prob:
        if entry < 0.5:
            pred.append(0)
        else:
            pred.append(1)
    return pred

# define the number of segments in a sequence
def get_number_of_segments(labels):
    count_segments = 0
    segment_sizes = []

    last_frame = 1
    for i in range(len(labels)):
        # start of a new segment: add to count and start measuring size
        if labels[i] == 0 and last_frame == 1:
            count_segments += 1
            last_frame = 0
            curr_segment_size = 1
        # continuing of a segment: add to size measuring
        elif labels[i] == 0 and last_frame == 0:
            curr_segment_size += 1
        # end of a segment: append to list of sizes
        elif labels[i] == 1 and last_frame == 0:
            last_frame = 1
            segment_sizes.append(curr_segment_size)
            curr_segment_size = 0
    
    return count_segments, segment_sizes

# compute IoU
def compute_iou(y_true, y_pred, ref=0):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    intersection = np.sum((y_true == ref) & (y_pred == ref))
    union = np.sum(((y_true == ref) | (y_pred == ref)))
    iou = float(intersection) / float(union)
    return iou

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

# get dataset
train, dev, test = get_datasets()

# get model
best_model = load_model(FLAGS.model_path, compile=False)
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
best_model.compile(
    loss=weighted_binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate),
    metrics=['accuracy', precision, recall],
    run_eagerly=True
)

# get IoU and % of segments
iou = []
sequence_ratio = []
for (input, label) in test:
    label = label.numpy().squeeze()
    prob = best_model.predict(input, verbose=0).squeeze()
    prediction = get_prediction(prob)
    
    # get IoU
    iou_value = compute_iou(y_true=label, y_pred=prediction, ref=0)
    iou.append(iou_value)
    
    # get % of segments
    original_sequences, _ = get_number_of_segments(label)
    predicted_sequences, _ = get_number_of_segments(prediction)
    sequence_ratio.append(predicted_sequences/original_sequences)

# compute metrics
result_test = best_model.evaluate(test, return_dict=True, verbose=0)
result_test['f1'] = safe_div(2*result_test['precision']*result_test['recall'],result_test['precision']+result_test['recall'])
result_test['iou'] = sum(iou)/len(iou)
result_test['% of segments'] = sum(sequence_ratio)/len(sequence_ratio)
print(result_test)
