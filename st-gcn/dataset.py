import glob
import os
import json
import numpy as np
import pandas as pd

def get_data_tensor(skel_folder, label_folder, split_file, split, save_folder, use_legs_feet=False, width=1280, height=720):
    """
    Input:
        skel_folder: The folder that have one subfolder for each video, containing the json files for each frame
        annot_folder: The folder that has the .txt label files
        split_file: The file that says which video is part of each split of the data
        split: The split that should be processed by the function
        save_folder: The folder to save results
        width: The width of the video, for data normalization
        height: The height of the video, for data normalization

    Output:
        A numpy array of shape (N, C, T, V, M), where:
            - N is the number of videos
            - C is the number of channels (in this case, 3 - x coordinate, y coodinate, and confidence)
            - T is the lenght of the input sequence (total number of frames)
            - V is the number of graph nodes (number of keypoints)
            - M is the number of instances in a frame (in this case, 1 - only one person per frame)
    """

    # define save folder for labels
    index_labels_folder = os.path.join(save_folder, split, 'labels')
    if not os.path.exists(index_labels_folder):
        os.makedirs(index_labels_folder)

    # define save folder for data
    split_save_folder = os.path.join(save_folder, split)
    if not os.path.exists(split_save_folder):
        os.makedirs(split_save_folder)

    # create mapping dict
    mapping = dict()

    # define skel
    keypoint_types = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
    body_points = {'pose_keypoints_2d': 25,
                    'face_keypoints_2d': 70,
                    'hand_left_keypoints_2d': 21,
                    'hand_right_keypoints_2d': 21}
    total_keypoints = 137

    # get the videos in the split
    split_csv = pd.read_csv(split_file)
    selected_split_names = split_csv.loc[split_csv['split']==split,'video'].tolist()
    total = len(selected_split_names)

    ## read each video
    i = 0
    for video in selected_split_names:

        # list all json files
        json_files = sorted(glob.glob(os.path.join(skel_folder,video,'*.json')))
        total_frames = len(json_files)

        # create empty numpy array
        if use_legs_feet:
            skel_data = np.empty(shape=(2, total_frames, total_keypoints, 1), dtype=np.float32)
        else:
            skel_data = np.empty(shape=(2, total_frames, total_keypoints-10, 1), dtype=np.float32) # remove legs and feet keypoints

        # get data from each file
        frame = 0
        for json_path in json_files:

            # read json
            file = open(json_path)
            data = json.load(file)

            # get skel data
            skel = data['people'][0]

            # get all skel values
            count = 0
            for body_part in keypoint_types:
                num_points = body_points[body_part]
                for point in range(num_points):
                    # if not using legs and feet keypoints
                    if (not use_legs_feet) and (body_part=='pose_keypoints_2d'):
                        if point in (10,11,13,14,19,20,21,22,23,24): # legs and feet keypoints
                            pass
                        else:
                            coord = point*3
                            if len(skel[body_part]) > 0:
                                skel_data[0, frame, count, 0] = (skel[body_part][coord]/width) # first coordinate
                                skel_data[1, frame, count, 0] = (skel[body_part][coord+1]/height) # second coordinate
                            # if data is missing, fill with zeros
                            else:
                                skel_data[0, frame, count, 0] = 0 # first coordinate
                                skel_data[1, frame, count, 0] = 0 # second coordinate
                            count = count + 1
                    # if using all the keypoints
                    else:
                        coord = point*3
                        if len(skel[body_part]) > 0:
                            skel_data[0, frame, count, 0] = (skel[body_part][coord]/width) # first coordinate
                            skel_data[1, frame, count, 0] = (skel[body_part][coord+1]/height) # second coordinate
                        # if data is missing, fill with zeros
                        else:
                            skel_data[0, frame, count, 0] = 0 # first coordinate
                            skel_data[1, frame, count, 0] = 0 # second coordinate
                        count = count + 1
            frame = frame + 1

        # save array with index ordering
        np.save(os.path.join(split_save_folder,f'{i}.npy'), skel_data)

        # save labels with index ordering
        labels = np.loadtxt(os.path.join(label_folder,f'{video}.txt')).astype(int)
        np.save(os.path.join(index_labels_folder,f'{i}.npy'), labels)

        # add mapping from index to video name
        mapping[i] = video

        i = i+1
        print(f'    Done {i}/{total}')    

    # save mapping file
    mapping_path = os.path.join(save_folder,f'{split}_mapping.txt')
    with open(mapping_path, 'w') as file:
        file.write(json.dumps(mapping))

    print(f'    Data files saved in {split_save_folder}')
    print(f'    Label files saved in {index_labels_folder}')
    print(f'    Mapping json saved in {mapping_path}\n')


## TO DO: Add these as arguments for this script
skel_folder = '/media/jessica/Storage2/libraset/scenes_skel'
label_folder = '/media/jessica/Storage2/libraset/scenes'
save_folder = './data/no_legs_and_feet/'
use_legs_feet = False
split_file = './split/split.csv'

for split in ['dev','test','train']:
    print(f'Preparing data from {split} split...')
    get_data_tensor(skel_folder, label_folder, split_file, split, save_folder, use_legs_feet)