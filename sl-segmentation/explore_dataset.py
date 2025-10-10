import sys
from args import FLAGS
from dataset import get_datasets
import numpy as np

FLAGS(sys.argv)

train, dev, test = get_datasets()

## train dataset
one_train = 0
zero_train = 0
prop_train = []
count = 0
print('Processing train dataset...')
for (input, label) in train:
    ones = np.sum(label.numpy())
    zeros = (label.shape[1] - ones)
    one_train = one_train + ones
    zero_train = zero_train + zeros
    prop_train.append(float(ones)/(ones+zeros))
    count = count+1
    # break if all observations were considered
    if count >= 212:
        break

print('Train dataset:')
print(f' Ones: {one_train}')
print(f' Zeros: {zero_train}')
print(f' Prop: {float(one_train)/(one_train+zero_train):.4f}')
print(f' Per entry prop: mean {np.mean(np.array(prop_train)):.4f}, std {np.std(np.array(prop_train)):.4f}, min {np.min(np.array(prop_train)):.4f}, median {np.median(np.array(prop_train)):.4f}, max {np.max(np.array(prop_train)):.4f}')
print('')

one_dev = 0
zero_dev = 0
prop_dev = []
print('Processing dev dataset...')
for (input, label) in dev:
    ones = np.sum(label.numpy())
    zeros = (label.shape[1] - ones)
    one_dev = one_dev + ones
    zero_dev = zero_dev + zeros
    prop_dev.append(float(ones)/(ones+zeros))

print('Dev dataset:')
print(f' Ones: {one_dev}')
print(f' Zeros: {zero_dev}')
print(f' Prop: {float(one_dev)/(one_dev+zero_dev):.4f}')
print(f' Per entry prop: mean {np.mean(np.array(prop_dev)):.4f}, std {np.std(np.array(prop_dev)):.4f}, min {np.min(np.array(prop_dev)):.4f}, median {np.median(np.array(prop_dev)):.4f}, max {np.max(np.array(prop_dev)):.4f}')
print('')

one_test = 0
zero_test = 0
prop_test = []
print('Processing test dataset...')
for (input, label) in test:
    ones = np.sum(label.numpy())
    zeros = (label.shape[1] - ones)
    one_test = one_test + ones
    zero_test = zero_test + zeros
    prop_test.append(float(ones)/(ones+zeros))

print('Test dataset:')
print(f' Ones: {one_test}')
print(f' Zeros: {zero_test}')
print(f' Prop: {float(one_test)/(one_test+zero_test):.4f}')
print(f' Per entry prop: mean {np.mean(np.array(prop_test)):.4f}, std {np.std(np.array(prop_test)):.4f}, min {np.min(np.array(prop_test)):.4f}, median {np.median(np.array(prop_test)):.4f}, max {np.max(np.array(prop_test)):.4f}')
print('')

print('Total dataset:')
print(f' Ones: {one_train + one_dev + one_test}')
print(f' Zeros: {zero_train + zero_dev + zero_test}')
print(f' Prop: {float(one_train + one_dev + one_test)/(one_train + one_dev + one_test + zero_train + zero_dev + zero_test):.4f}')
prop = prop_train + prop_dev + prop_test
print(f' Per entry prop: mean {np.mean(np.array(prop)):.4f}, std {np.std(np.array(prop)):.4f}, min {np.min(np.array(prop)):.4f}, median {np.median(np.array(prop)):.4f}, max {np.max(np.array(prop)):.4f}')
