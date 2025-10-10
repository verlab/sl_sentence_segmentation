import glob
import os
import json
import numpy as np
from pose_format.utils.openpose import load_openpose_directory

# base folder
base_folder = './selected_videos_pose/'
save_folder = './selected_videos_normalized/'

# list directories in base folder
subfolders = glob.glob(base_folder+'*/')

# process data
total_folders = len(subfolders)
count = 0
mapping = dict()
for directory in subfolders:
    # read pose
    pose = load_openpose_directory(directory, fps=30, width=1920, height=1080)

    # normalize so that shoulders have distance = 1
    pose.normalize(pose.header.normalization_info(
        p1=("pose_keypoints_2d", "RShoulder"),
        p2=("pose_keypoints_2d", "LShoulder")
    ))

    # reorganize the numpy arrays
    data = pose.body.data.data
    legs_and_feet_coordinates = (10,11,13,14,19,20,21,22,23,24) # remove legs and feet
    data_no_legs_and_feet = np.delete(data, legs_and_feet_coordinates, axis=2)

    # confidence = pose.body.confidence
    # confidence = np.expand_dims(confidence, axis=-1)
    # confidence_no_legs_and_feet = np.delete(confidence, legs_and_feet_coordinates, axis=2)

    # concatenated_data = np.concatenate((data_no_legs_and_feet, confidence_no_legs_and_feet), axis=-1)
    # concatenated_data = np.expand_dims(concatenated_data, axis=0)
    concatenated_data = np.expand_dims(data_no_legs_and_feet, axis=0)
    final_data = np.transpose(concatenated_data, (0, 4, 1, 3, 2))
    final_data = np.squeeze(final_data, axis=0)

    # save array
    folder_name = directory.split('/')[-2]
    mapping[count] = folder_name
    np.save(os.path.join(save_folder, f'{count}.npy'), final_data)
    print(f'({count+1}/{total_folders}) Folder {folder_name} finished')
    count = count+1

mapping_path = os.path.join(save_folder, 'mapping.txt')
with open(mapping_path, 'w') as file:
    file.write(json.dumps(mapping))

print(f'\n\nData files saved in {save_folder}')
print(f'Mapping json saved in {mapping_path}\n')
