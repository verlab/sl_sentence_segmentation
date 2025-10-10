# Sign Language Video Segmentation

Implementation and evaluation of models for automatic segmentation of continuous sign language into sentence units.

## Repository Organization

- `sl-segmentation`: Implementation of the segmentation model based in optical flow and experiments.
- `st-gcn`: Implementation of a segmentation model that uses spatio-temporal graph convolutional networks for processing pose sequences and experiments.

## `sl-segmentation`
**Real-Time Sign Language Detection using Human Pose Estimation ([Code](https://github.com/google-research/google-research/tree/master/sign_language_detection))**

A TensorFlow implementation of the model presented in ["Real-Time Sign Language Detection using Human Pose Estimation"](https://slrtp.com/papers/full_papers/SLRTP.FP.04.017.paper.pdf), published at SLRTP 2020.

## `st-gcn`
**Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition ([Code](https://github.com/open-mmlab/mmskeleton))**

A PyTorch implementation of the model presented in ["Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition"](https://cdn.aaai.org/ojs/12328/12328-13-15856-1-2-20201228.pdf), adapted for the task of sign language video segmentation, published at AAAI-18. The applied approach is similar to the one used in ["Automatic Segmentation of Sign Language into Subtitle-Units"](https://slrtp.com/papers/full_papers/SLRTP.FP.01.011.paper.pdf) ([Code](https://github.com/hannahbull/sign_language_segmentation/tree/master)), published at SLRTP 2020.

## Tools
Extraction of skeletons using Mediapipe and conversion to OpenPose format: [Code](https://github.com/verlab/captar-libras-mediapipe)

