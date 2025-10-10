import os
import torch
from feeder.feeder import Feeder
from backbone.st_gcn_aaai18 import ST_GCN_18
from torchmetrics import Accuracy, Precision, Recall
import argparse
import numpy as np

def custom_weighted_cross_entropy(pred_prob, true_labels):
  weights = torch.where(true_labels==1, torch.tensor(class_weight), torch.tensor(1.0))
  loss = torch.nn.functional.binary_cross_entropy(pred_prob, true_labels.float(), weight=weights)
  return loss
  
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

# using same arguments as train.py code to make things easier
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, help='Path to folder containing train .npy files', default='./data/no_legs_and_feet/train/')
    parser.add_argument('--dev_path', type=str, help='Path to folder containing dev .npy files', default='./data/no_legs_and_feet/dev/')
    parser.add_argument('--test_path', type=str, help='Path to folder containing test .npy files', default='./data/no_legs_and_feet/test/')
    parser.add_argument('--in_channels', type=int, help='Number of channels in input data', default=2)
    parser.add_argument('--strategy', type=str, help='Strategy for graph partition', default='spatial')
    parser.add_argument('--which_keypoints', type=str, help='Specify "body", "headbody", "bodyhands", "headface" or "full".', default='body')
    parser.add_argument('--legs_and_feet', type=bool, help='If the original data includes hands and feet keypoints', default=False)
    parser.add_argument('--class_weight', type=float, help='Weight of the minority class', default=5.0)
    parser.add_argument('--num_layers', type=int, help='Number of LSTM layers', default=1)
    parser.add_argument('--st_gcn_layers', type=int, help='Number of ST-GCN layers', default=10)
    parser.add_argument('--hidden_size', type=int, help='Hidden size for LSTM', default=64)
    parser.add_argument('--bidirectional', type=bool, help='If the LSTM layers are bidirectional', default=False)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs', default=50)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.001)
    parser.add_argument('--model_path', type=str, help='Path to save the model')
    parser.add_argument('--batch_size', type=int, help='Batch size for training', default=1)
    parser.add_argument('--test_batch_size', type=int, help='Batch size for testing', default=1)

    args = parser.parse_args()
    args_list = vars(args)
    
    return args_list

## get arguments
argument_list = get_arguments()

# data
train_path = argument_list['train_path']
dev_path = argument_list['dev_path']
test_path = argument_list['test_path']

legs_and_feet = argument_list['legs_and_feet']

if argument_list['which_keypoints'] == 'full':
    # full pose with legs and feet
    if legs_and_feet:
        layout = 'openpose (137)'
    # full pose without legs and feet
    else:
        layout = 'openpose (127, no legs and feet)'
    keypoints_group = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
elif argument_list['which_keypoints'] == 'body':
    # only body keypoints with legs and feet
    if legs_and_feet:
        layout = 'openpose (25, body only)'
    # only body keypoints without legs and feet 
    else:
        layout = 'openpose (15, body only, no legs and feet)'
    keypoints_group = ['pose_keypoints_2d']
# head + body keypoints without legs and feet
elif argument_list['which_keypoints'] == 'headbody' and not legs_and_feet:
    layout = 'openpose (85, body and face, no legs and feet)'  
    keypoints_group = ['pose_keypoints_2d', 'face_keypoints_2d']
# hands + body keypoints without legs and feet
elif argument_list['which_keypoints'] == 'bodyhands' and not legs_and_feet:
    layout = 'openpose (57, body and hands, no legs and feet)'
    keypoints_group = ['pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
# face + hands keypoints
elif argument_list['which_keypoints'] == 'headface' and not legs_and_feet:
    layout = 'openpose (112, face and hands)'
    keypoints_group = ['face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
# if combination is not supported by code, exit
else:
    print(f'Keypoint group "{argument_list["which_keypoints"]}" with legs_and_feet = {legs_and_feet} does not exist.')
    exit()

in_channels = argument_list['in_channels']
graph_cfg = {'strategy': argument_list['strategy'],
             'layout': layout}

class_weight = argument_list['class_weight']

# model
hidden_size = argument_list['hidden_size']
num_layers = argument_list['num_layers']
bidirectional = argument_list['bidirectional']
st_gcn_layers = argument_list['st_gcn_layers']

# training
num_epochs = argument_list['num_epochs']
learning_rate = argument_list['learning_rate']
model_path = argument_list['model_path']
batch_size = argument_list['batch_size']
test_batch_size = argument_list['test_batch_size']

## device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## data loader
test = torch.utils.data.DataLoader(
                dataset=Feeder(
                    data_path=test_path,
                    label_path=os.path.join(test_path,'labels'),
                    keypoints_group=keypoints_group,
                    legs_and_feet=legs_and_feet
                ),
                batch_size=test_batch_size,
                shuffle=False)

## loss (to load the model)
loss_fn = custom_weighted_cross_entropy

## metrics
accuracy = Accuracy(task='binary').to(device)
precision = Precision(task='binary').to(device)
recall = Recall(task='binary').to(device)

## evaluate the best model on test set
best_model = ST_GCN_18(
    in_channels=in_channels,
    graph_cfg=graph_cfg,
    hidden_size=hidden_size,
    num_layers=num_layers,
    st_gcn_layers=st_gcn_layers,
    bidirectional=bidirectional
).to(device)
best_model.load_state_dict(torch.load(model_path))

best_model.eval()
test_loss = 0.0
accuracy.reset()
precision.reset()
recall.reset()
iou = []
sequence_ratio = []
with torch.no_grad():
    for inputs, labels in test:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = best_model(inputs)
        outputs = outputs.to(device)
        loss = loss_fn(outputs, labels)
        test_loss += loss.item()
        
        # Calculate metrics
        accuracy.update(outputs, labels)
        precision.update(outputs, labels)
        recall.update(outputs, labels)
        
        # iou
        labels_cpu = labels.cpu().numpy().squeeze()
        pred_array = get_prediction(outputs.cpu().numpy().squeeze())
        iou_value = compute_iou(y_true=labels_cpu, y_pred=pred_array, ref=0)
        iou.append(iou_value)
        
        # sequence ratio
        original_sequences, _ = get_number_of_segments(labels.cpu().numpy().squeeze())
        predicted_sequences, _ = get_number_of_segments(pred_array)
        sequence_ratio.append(predicted_sequences/original_sequences)        

# Compute final metrics on test set
test_accuracy = accuracy.compute()
test_precision = precision.compute()
test_recall = recall.compute()
test_iou = sum(iou)/len(iou)
test_sequence_ratio = sum(sequence_ratio)/len(sequence_ratio)
test_f1 = safe_div(2*test_precision*test_recall, test_precision+test_recall)
print(f'\nFinal Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, IoU (sentences): {test_iou:.4f}, % of segments: {test_sequence_ratio:.4f}')
