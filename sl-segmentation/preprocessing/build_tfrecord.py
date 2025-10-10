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

''' Creates tfrecord file for inference (only have pose data)'''

import argparse
from glob import glob
import os
import numpy as np
import tensorflow as tf
import json
from build_tfrecord_train import get_json_data

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--skel', type=str, required=True, help = 'Folder with skeleton information (JSON files)')
    parse.add_argument('--type_skel', type=str, required=True, help = 'Type of skeleton data ("OpenPose" or "DGS")')
    parse.add_argument('--fps', type=int, required=True, help='FPS of the video')
    parse.add_argument('--output', type=str, required=True, help='Name of resulting the tfrecord file', default='dataset')
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    # check output file name
    if args.output[-9:] == '.tfrecord':
        file_name = args.output
    else:
        file_name = args.output + '.tfrecord'

    # get list of files
    if args.type_skel == 'OpenPose':
        videos_list = glob(args.skel + '*/')
    elif args.type_skel == 'DGS':
        videos_list = glob(args.skel + '*.json')

    # create tfrecord file for training
    count = 0
    video_names = dict()
    print('Processing...')
    with tf.io.TFRecordWriter(file_name) as writer:
        for video in videos_list:

            # save video name and id
            video_names[count] = video
            print(f'\nProcessing video {count}: {video}')

            # get skeleton data
            skel_data, conf_data, info = get_json_data(video, json_type=args.type_skel)

            # organize data for each camera
            for i in range(len(skel_data)):
                data = tf.io.serialize_tensor(tf.convert_to_tensor(skel_data[i])).numpy()
                confidence = tf.io.serialize_tensor(tf.convert_to_tensor(conf_data[i])).numpy()
    
                features = {
                    'fps': tf.train.Feature(int64_list=tf.train.Int64List(value=[args.fps])),
                    'pose_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data])),
                    'pose_confidence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[confidence]))
                }

                # write to file
                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())
                    
            count = count + 1
            print(f'{round(100*count/len(videos_list))}% processed.')
    
    # save video names to txt file
    with open(f'{args.output}/video_names.txt', 'w') as f:
        for key, value in video_names.items():
            f.write(f'{key},"{value}"\n')

    print(f'tfrecord created: {file_name}')
    print(f'Video names saved to {args.output}/video_names.txt')

