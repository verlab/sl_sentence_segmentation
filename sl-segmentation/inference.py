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
from tensorflow import keras

from args import FLAGS
import sys
FLAGS(sys.argv)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime


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
  
minimum_fps = tf.constant(1, dtype=tf.float32)

def load_datum(tfrecord_dict):
    """Convert tfrecord dictionary to tensors."""
    pose_body = TensorflowPoseBody.from_tfrecord(tfrecord_dict)
    pose = Pose(header=get_openpose_header(), body=pose_body)
    fps = pose.body.fps
  
    # Get only relevant input components
    pose = pose.get_components(FLAGS.input_components)
  
    return {
        "fps": pose.body.fps,
        "pose_data_tensor": pose.body.data.tensor,
        "pose_data_mask": pose.body.data.mask,
        "pose_confidence": pose.body.confidence,
    }
  
def process_datum(datum, augment=False):
    """Prepare every datum to be an input-output pair for training / eval.
    Supports data augmentation only including frames dropout.
    Frame dropout affects the FPS, which does change the optical flow.
    Args:
        datum (Dict[str, tf.Tensor]): a dictionary of tensors loaded from the
          tfrecord.
        augment (bool): should apply data augmentation on the datum?
    Returns:
       src tensors
    """
    masked_tensor = MaskedTensor(
        tensor=datum["pose_data_tensor"], mask=datum["pose_data_mask"])
    pose_body = TensorflowPoseBody(
        fps=datum["fps"], data=masked_tensor, confidence=datum["pose_confidence"])
    pose = Pose(header=get_openpose_header(), body=pose_body)

    fps = pose.body.fps
    flow = optical_flow(pose.body.data, fps)

    return {"src": flow}

def batch_dataset(dataset, batch_size):
    """Batch and pad a dataset."""
    dataset = dataset.padded_batch(
        batch_size, padded_shapes={'src':[None, None]})

    return dataset.map(prepare_io)
  
def prepare_io(datum):
    """Convert dictionary into input tuple for Keras."""
    src = datum["src"]
    return src
  
def recover(_, y):
    return y

def get_segments(y, tolerance):
    """
    Extract the indexes of the segments of 1s in a sequence y.

    Inputs:
        y: numpy array with dimensions (sequence_lenght,).
        tolerance: is the number (int) of entries with the same value (0 or 1) to
            consider as start or end of a segment.
    
    Output:
        Returns a list with start and finish indexes of the segments in the sequence.
    """
    count_1 = 0
    count_0 = 0
    is_segment = False
    segments_list = []
    start = None
    finish = None
    for i in range(y.shape[0]):
        # if between segments
        if is_segment is False:
            if y[i] == 1:
                if (i==0) or (y[i-1]==1):
                    count_1 = count_1+1
                else:
                    count_1 = 1
            else:
                count_1 = 0
            if count_1 == tolerance:
                start = i-tolerance+1
                is_segment = True
                count_0 = 0
                count_1 = 0
        # if in the middle of a segment
        elif is_segment is True:
            if y[i] == 0:
                if (i==0) or (y[i-1]==0):
                    count_0 = count_0+1
                else:
                    count_0 = 1
            else:
                count_0 = 0
            if count_0 == tolerance:
                finish = i-tolerance
                is_segment = False
                segments_list.append([start, finish])
                count_0 = 0
                count_1 = 0
        
        # if it is the last frame and is in the middle of a segment, end segment
        if (i == (y.shape[0]-1)) and (is_segment is True):
            finish = i
            is_segment = False
            segments_list.append([start, finish])
            count_0 = 0
            count_1 = 0

    return segments_list
    
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # read tfrecord file
    dataset = tf.data.TFRecordDataset(filenames=[FLAGS.dataset_path])
    features = TF_POSE_RECORD_DESCRIPTION
    dataset = dataset.map(lambda serialized: tf.io.parse_single_example(serialized, features))
    dataset = dataset.enumerate().map(recover)
    dataset = dataset.map(load_datum)
    dataset = dataset.map(process_datum)
    dataset = batch_dataset(dataset, FLAGS.test_batch_size)
    
    # load model
    model = keras.models.load_model('results/models/model_16.h5', compile=False) # change model here
    
    # make prediction
    predictions = model.predict(dataset.cache())

    # generate outputs
    # for i in range(len(predictions)):
    for i, pred in enumerate(predictions):
        # pred = predictions[i]
        pred = pred.numpy()
        output = np.zeros(shape=pred.shape)
        for frame in range(pred.shape[0]):
            output[frame] = pred[frame]
        pred = output
    
        # convert probabilities to 0 or 1
        class_pred = np.where(pred>0.5, 1, 0)
        segments = get_segments(class_pred, 5)

        # save output
        np.savetxt(f'inference_new_videos/model_16/output_{i}.csv', class_pred, delimiter=',')

        # # draw prediction
        # seconds = np.array(range(len(pred)))/50
        # plt.figure(figsize=(13,3))
        # plt.plot(seconds, pred, color = 'black', alpha = 0.1)
        # for entry in segments:
        #     start = entry[0]/50
        #     finish = entry[1]/50
        #     plt.gca().add_patch(mpatches.Rectangle((start, 0), (finish-start), 1, alpha=0.5, facecolor='green'))
        # plt.axhline(y=0.5, linestyle='--', color='black')
        # plt.savefig(f'inference_new_videos/model_19/pred_{date_time}_{i}.png')
        # plt.close()

        print(f'Processed video {i+1}')   
