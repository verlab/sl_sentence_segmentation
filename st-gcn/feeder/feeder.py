## This code is adapted from https://github.com/yysijie/st-gcn/blob/master/feeder/feeder.py

# sys
import os
import glob
import numpy as np

# torch
import torch
import torch.nn.functional as F

class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to folder with data labels in .npy format and index ordering
        keypoints_group: the group of keypoints to be considered
        legs_and_feet: if the legs and feet keypoints are considered for body
    """

    def __init__(self,
                 data_path,
                 keypoints_group,
                 legs_and_feet,
                 label_path=None,
                 debug=False):
        
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.keypoints_group = keypoints_group
        self.legs_and_feet = legs_and_feet

    def __len__(self):
        return len(glob.glob(os.path.join(self.data_path,'*.npy')))

    def __getitem__(self, index):
        # get data
        data_numpy = np.load(os.path.join(self.data_path, f'{index}.npy'), allow_pickle=True)
        if self.label_path is not None:
            label = np.load(os.path.join(self.label_path, f'{index}.npy'), allow_pickle=True)
        else:
            label = np.zeros(data_numpy.shape[1])
        # pose only
        if self.keypoints_group == ['pose_keypoints_2d']:
            if self.legs_and_feet:
                pose_only = data_numpy[:,:,:25,:]
            else:
                pose_only = data_numpy[:,:,:15,:]
            return pose_only, label
        
        # pose and face
        elif self.keypoints_group == ['pose_keypoints_2d', 'face_keypoints_2d']:
            if self.legs_and_feet:
                pose_only = data_numpy[:,:,:95,:]
            else:
                pose_only = data_numpy[:,:,:85,:]
            return pose_only, label
        
        # pose and hands
        elif self.keypoints_group == ['pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
            if self.legs_and_feet:
                indices = np.concatenate((np.arange(0, 25), np.arange(95, 137)))
                pose_and_hands = data_numpy[:,:,indices,:]
            else:
                indices = np.concatenate((np.arange(0, 15), np.arange(85, 127)))
                pose_and_hands = data_numpy[:,:,indices,:]
            return pose_and_hands, label
        
        # full
        elif self.keypoints_group == ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
            return data_numpy, label

        # # hands only
        # elif self.keypoints_group == ['hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
        #     if self.legs_and_feet:
        #         hands_only = data_numpy[:,:,95:,:]
        #     else:
        #         hands_only = data_numpy[:,:,85:,:]
        #     return hands_only, label

        # face and hands
        elif self.keypoints_group == ['face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
            if self.legs_and_feet:
                face_and_hands = data_numpy[:,:,25:,:]
            else:
                face_and_hands = data_numpy[:,:,15:,:]
            return face_and_hands, label
            
