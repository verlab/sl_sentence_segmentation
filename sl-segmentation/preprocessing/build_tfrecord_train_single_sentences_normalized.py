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

''' Creates tfrecord file for training (only single sentence verses in JW-Bible-Libras) '''

import argparse
from glob import glob
import os
import numpy as np
import tensorflow as tf
import json
import glob
import pandas as pd

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--skel', type=str, required=True, help = 'Folder with skeleton information (JSON files)')
    parse.add_argument('--annot', type=str, required=True, help = 'Folder with txt annotation files')
    parse.add_argument('--fps', type=int, required=True, help='FPS of the videos')
    parse.add_argument('--output', type=str, required=True, help='Name of the resulting tfrecord file', default='dataset')
    args = parse.parse_args()
    return args

def normalize_openpose_pose(json_path, output_path):
    # Load JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract keypoints (assuming only one person)
    people = data.get("people", [])
    if not people:
        print("No pose data found in the JSON file.")
        return
    
    body_keypoints = np.array(people[0]["pose_keypoints_2d"]).reshape(-1, 3)  # Reshape to (N, 3)
    left_hand_keypoints = np.array(people[0]["hand_left_keypoints_2d"]).reshape(-1, 3)
    right_hand_keypoints = np.array(people[0]["hand_right_keypoints_2d"]).reshape(-1, 3)
    face_keypoints = np.array(people[0]["face_keypoints_2d"]).reshape(-1, 3)

    if len(body_keypoints) > 0:
        # Get shoulders
        LEFT_SHOULDER = 5  # OpenPose index for left shoulder
        RIGHT_SHOULDER = 2  # OpenPose index for right shoulder

        left_shoulder = body_keypoints[LEFT_SHOULDER][:2]  # (x, y)
        right_shoulder = body_keypoints[RIGHT_SHOULDER][:2]  # (x, y)

        # Compute shoulder distance
        shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)

        # Normalize all keypoints by dividing by shoulder distance
        body_keypoints[:, :2] /= shoulder_distance
        
        # check if hands and face keypoints were detected. if not, just use the empty list again
        if len(left_hand_keypoints) > 0:
            left_hand_keypoints[:, :2] /= shoulder_distance
        
        if len(right_hand_keypoints) > 0:
            right_hand_keypoints[:, :2] /= shoulder_distance
        
        if len(face_keypoints) > 0:
            face_keypoints[:, :2] /= shoulder_distance

        # Save the normalized keypoints back
        people[0]["pose_keypoints_2d"] = body_keypoints.flatten().tolist()
        people[0]["hand_left_keypoints_2d"] = left_hand_keypoints.flatten().tolist()
        people[0]["hand_right_keypoints_2d"] = right_hand_keypoints.flatten().tolist()
        people[0]["face_keypoints_2d"] = face_keypoints.flatten().tolist()
        data["people"] = people
    
    else:
        people[0]["pose_keypoints_2d"] = []
        people[0]["hand_left_keypoints_2d"] = []
        people[0]["hand_right_keypoints_2d"] = []
        people[0]["face_keypoints_2d"] = []
        data["people"] = people

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

def get_json_data(scene, initial_frame, final_frame, resolution=(1280,720)):
    '''
    Organizes human pose keypoints data in numpy arrays. It assumes only one person in each video.

    Input:
        scene (str): Path to folder of the scene the verse is in with json files (OpenPose) with body, face, and hand keypoints.
        initial_frame (int): The first frame of the scene that includes the verse of interest.
        final_frame (int): The last frame of the scene that includes the verse of interest.
        resolution (tuple): (Optional) The resolution of the videos processed for keypoint normalization.

    Output:
        skel_data (list of np.arrays): List of arrays of shape (#frames, 1, 137, 2) with coordinates for each keypoint.
        conf_data (list of np.arrays): List of arrays of shape (#frames, 1, 137) with human pose confidence estimation for each keypoint.
    '''
    
    # list number of keypoints for body part
    keypoint_types = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
    body_points = {'pose_keypoints_2d': 25,
                   'face_keypoints_2d': 70,
                   'hand_left_keypoints_2d': 21,
                   'hand_right_keypoints_2d': 21}
    total_keypoints = 137

    # get frames width
    width = resolution[0]
    height = resolution[1]

    # list all json files
    all_json_files = glob.glob(os.path.join(scene,'*.json'))

    # get only the json files in the scene interval
    json_files = sorted([file for file in all_json_files if initial_frame <= int(os.path.basename(file).split('_')[1].split('.')[0]) <= final_frame])
    total_frames = len(json_files)
    print(f'    Frames: {total_frames}')

    # create output list
    skel_list = []
    conf_list = []
    info = []

    # create empty numpy array
    skel_data = np.empty(shape=(total_frames, 1, total_keypoints, 2), dtype=np.float32)
    conf_data = np.empty(shape=(total_frames, 1, total_keypoints), dtype=np.float32)

    # for each frame (json file)
    frame = 0
    for file_name in json_files:
        # normalize pose in temp file
        normalize_openpose_pose(file_name, 'temp.json')

        # get json data
        file = open('temp.json')
        data = json.load(file)
        skel = data['people'][0] # only first person

        # get all skeleton values
        count = 0
        for body_part in keypoint_types:
            num_points = body_points[body_part]
            for point in range(num_points):
                coord = point*3
                if len(skel[body_part]) > 0:
                    skel_data[frame, 0, count, 0] = (skel[body_part][coord]) # first coordinate
                    skel_data[frame, 0, count, 1] = (skel[body_part][coord+1]) # second coordinate
                    conf_data[frame, 0, count] = skel[body_part][coord+2] # confidence
                # if data is missing, fill with zeros
                else:
                    skel_data[frame, 0, count, 0] = 0 # first coordinate
                    skel_data[frame, 0, count, 1] = 0 # second coordinate
                    conf_data[frame, 0, count] = 0 # confidence
                count = count + 1
        frame = frame + 1
        os.remove('temp.json')

    # add to output list
    info.append(('cam', total_frames))
    skel_list.append(skel_data)
    conf_list.append(conf_data)

    return skel_list, conf_list, info


if __name__ == '__main__':
    args = get_args()

    # check output file name
    if args.output[-9:] == '.tfrecord':
        file_name = args.output
    else:
        file_name = args.output + '.tfrecord'

    # read split csv
    dev_csv = pd.read_csv('./data/single_sentence_verses/dev/frames.csv')
    test_csv = pd.read_csv('./data/single_sentence_verses/test/frames.csv')
    train_csv = pd.read_csv('./data/single_sentence_verses/train/frames.csv')

    # add split column and groups csvs
    dev_csv['split'] = 'dev'
    test_csv['split'] = 'test'
    train_csv['split'] = 'train'
    split_csv = pd.concat([train_csv, test_csv, dev_csv], axis=0)

    # create verses list
    verses_list = split_csv['verse_name'].tolist()

    # create tfrecord file for training
    count = 0
    print('Processing...')
    
    # create file with indexes and video names
    index_file = open(args.output + '.txt', 'w+')
    index_file.write('index;file_name\n')

    # create index list of splits
    dev_index_list = []
    test_index_list = []
    train_index_list = []
    
    with tf.io.TFRecordWriter(file_name) as writer:
        for verse in verses_list:
            print(f'\nVerse: {verse}')
            index_file.write(f'{count};{verse}\n')

            # save split
            row = split_csv[split_csv['verse_name']==verse]
            if row['split'].values[0] == 'dev':
                dev_index_list.append(count)
            elif row['split'].values[0] == 'test':
                test_index_list.append(count)
            elif row['split'].values[0] == 'train':
                train_index_list.append(count)

            # get skeleton data
            scene = os.path.join(args.skel, row['scene'].values[0])
            initial_frame = int(row['initial_frame'].values[0])
            final_frame = int(row['final_frame'].values[0])
            skel_data, conf_data, info = get_json_data(scene, initial_frame, final_frame)
            
            # get annotation data (from txt file with numpy array)
            video_name = row['scene'].values[0]
            annot_file = os.path.join(args.annot, video_name+'.txt')
            annot_data = [np.loadtxt(annot_file).astype(int)]
            final_frame_plus1 = final_frame + 1
            annot_data[0] = annot_data[0][initial_frame:final_frame_plus1] # only the frames in the verse interval
            print(f'    Annotations: {len(annot_data[0])}')
            print(f'    % of 1s: {round(100*np.sum((annot_data[0]))/len(annot_data[0]),2)}')
            error_list = []

            # organize data
            for i in range(len(skel_data)):
            	if i not in error_list:
                    data = tf.io.serialize_tensor(tf.convert_to_tensor(skel_data[i])).numpy()
                    confidence = tf.io.serialize_tensor(tf.convert_to_tensor(conf_data[i])).numpy()
                    is_signing = annot_data[i].tobytes()
    
                    features = {
                        'fps': tf.train.Feature(int64_list=tf.train.Int64List(value=[args.fps])),
                        'pose_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data])),
                        'pose_confidence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[confidence])),
                        'is_signing': tf.train.Feature(bytes_list=tf.train.BytesList(value=[is_signing]))
                    }

                    # write to file
                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(example.SerializeToString())
                    
            count = count + 1
            print(f'{round(100*count/len(verses_list))}% processed.')
    
    index_file.close()
    print(f'tfrecord created: {file_name}')

    # save split files
    np.savetxt('split/dev_single_sentences.csv', dev_index_list)
    np.savetxt('split/test_single_sentences.csv', test_index_list)
    np.savetxt('split/train_single_sentences.csv', train_index_list)
