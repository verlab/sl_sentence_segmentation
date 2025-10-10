import numpy as np
import matplotlib.pyplot as plt
from dataset import get_datasets
import sys
import os
from args import FLAGS

FLAGS(sys.argv)

# read datasets
train, dev, test = get_datasets()

# read predictions
pred_file = './results/predictions/model_9_test.npy'
predictions = np.load(pred_file, allow_pickle=True)

# get test labels into lists
labels = []
for (input, label) in test:
    labels.append(label)

# create folder for predictions
folder = 'results/plots/model_9'
if not os.path.exists(folder):
    os.makedirs(folder)

# make plots
for i in range(predictions.shape[0]):
    prob = np.array(predictions[i])
    true_labels = labels[i].numpy()
    true_labels = true_labels.reshape(true_labels.shape[1])
    x = range(len(prob))
    plt.figure(figsize = (20,4))
    plt.plot(x, true_labels, label='Ground Truth', linewidth=3)
    plt.plot(x, prob, label='Prediction', linestyle='--')
    plt.axhline(y=0.5, linestyle='--', color='gray')
    plt.legend()
    plt.savefig(os.path.join(folder,f'prediction_{i}.png'))
    plt.close()
    print(f'{i}/{predictions.shape[0]} done.')
