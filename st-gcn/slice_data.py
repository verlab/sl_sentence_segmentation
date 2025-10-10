import numpy as np
import glob
import os
from tqdm import tqdm

def create_folder(path):
    """
    Check if a path exists and, if not, creates it.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def slice_array(keypoints_array, labels_array, slice_size=1800, stride=900):
    """
    Slices the input array in the third dimension.

    Parameters:
    keypoints_array (numpy.ndarray): Input array of shape (C, T, V, M) for keypoints.
    labels_array (numpy.ndarray): Input array of shape (T,) for labels.
    slice_size (int): Size of each slice in the third dimension. Default is 1800.
    stride (int): Stride for the slices in the third dimension. Default is 900.

    Returns:
    List: List of slices.
    """
    # Ensure the input array has the correct number of dimensions
    assert keypoints_array.ndim == 4, "Input array must have shape (C, T, V, M)"
    
    # Calculate the number of slices we can take
    num_slices = (keypoints_array.shape[1] - slice_size) // stride + 1
    
    # List to hold slices
    slices = []
    labels = []
    
    # Loop to create slices
    for i in range(num_slices):
        start = i * stride
        if i == (num_slices-1):
            # on last slice, grab the end of the array
            end = keypoints_array.shape[1]
        else:
            # otherwise, grab slice_size next frames
            end = start + slice_size
        slices.append(keypoints_array[:, start:end, :, :])
        labels.append(labels_array[start:end])
    
    # Return list of slices
    return slices, labels

# apply operation to data
base_folder = './data/no_legs_and_feet_normalized/'
new_base_folder = './data/no_legs_and_feet_normalized_sliced_60s/'
#for split in ['test','dev','train']:
for split in ['train']:
    print(f'Processing files from {split}...')

    # get folder with the original data
    origin_folder = os.path.join(base_folder, split)

    # create new folder
    new_folder = os.path.join(new_base_folder, split)
    create_folder(os.path.join(new_folder,'labels'))

    # get files
    files_list = sorted(glob.glob(os.path.join(origin_folder, '*.npy')))
    count = 0
    for j in tqdm(range(len(files_list))):
        # get file
        file = files_list[j]

        # read array
        keypoints_array = np.load(file)

        # read labels
        id = file.split('/')[-1].split('.')[0]
        labels_array = np.load(os.path.join(base_folder, split, f'labels/{id}.npy'))
        
        # slice array
        slices, labels = slice_array(keypoints_array, labels_array, slice_size=1800, stride=900)

        # save arrays
        for i in range(len(slices)):
            new_slice = slices[i]
            new_labels = labels[i]
            np.save(os.path.join(new_folder, f'{count}.npy'), new_slice)
            np.save(os.path.join(new_folder,f'labels/{count}.npy'), new_labels)
            count = count+1
    
    print(f'Saved {count} files in {new_folder}\n')
