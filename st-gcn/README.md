# Automatic Segmentation of Sign Language into Subtitle-Units

[Paper](https://slrtp.com/papers/full_papers/SLRTP.FP.01.011.paper.pdf)

[Original Repo](https://github.com/hannahbull/sign_language_segmentation)

## Create Docker Image and Container

`docker build -t st-gcn .`

`docker run -it --gpus all [--arguments] st-gcn`

## Prepare the Dataset

1. Extract OpenPose poses from the video frames. The poses should be organized within the same folder, with each subfolder containing the JSON files for each frame.


```
skel
├── video_1
    ├── frame_00000.json
    ├── frame_00001.json
    ├── frame_00002.json
    ...
├── video_2
    ├── frame_00000.json
    ├── frame_00001.json
    ├── frame_00002.json
    ...
```

2. Create the label files. The files should have the same name as the folders identifying the videos, with a `.txt` extension (`video_1.txt`, `video_2.txt`, etc.). Each file contains one line per video frame. The value is 0 when the frame is part of a sentence, and 1 when the frame is at the boundary between two sentences. For example:

```
0
0
0
1
1
1
1
1
0
0
...
0
0
0
1
1
1
1
```

3. Create the split file, specifying which videos belong to the training, validation, and test datasets. The file is a CSV with the following format:

```
video,split
video_1,train
video_2,train
video_3,dev
video_4,test
...
```

4. Update the arguments in `dataset.py` and run the script.

```
skel_folder = './data/skel' # folder with poses
label_folder = './data/label' # folder with label files
save_folder = './data/model_data' # folder to save the numpy arrays
use_legs_feet = False # whether the numpy array should include keypoints for legs and feet (if false, removes 10 keypoints from the main pose)
split_file = './split/split.csv' # file with the data split
```


6. Run training with `python3 train.py [--arguments]`. The arguments are defined at the start of the script.
- The script saves the model at the path defined by `--model_path`. The saved model is the one with the best validation dataset performance during training.
- The model loss plot per epoch is saved in `./results/plot`.
- The script reports accuracy, precision, and recall calculated on class `1` for each epoch. The final result is computed using the best model and also includes the F1-score and % of segments (# of predicted sentences / # of true sentences).
- The IoU is calculated relative to sentences (how much the predicted sentences overlap with the ground truth sentences).
- Predicting a single frame `1` indicates the start or end of a sentence by default.

