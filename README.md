# Hand Gesture Tracking and Recognition by Fully-Convolutional Siamese Network

## Introduction
This repository contains the codes and results to track hand gesture and recognise on my own dataset. This project is the Pytorch implementation of Fully-Convolutional Siamese Network.  
The project is divided into four major parts : **Training, Evaluation, Tracking, Recognise.**

## Environment setup
Clone this repository :
```
git clone https://github.com/SuchibrataBhowmik/Hand-Gesture-Tracking-and-Recognise.git
```
```
cd Hand-Gesture-Tracking-and-Recognise
```
Please install related libraries on your environment before running this code :  
 ```
 pip3 install -r requirements.txt
```
This code tested on Intel® Xeon(R) CPU, 3.30GHz × 4 and GeForce RTX 2080 GPU.  
If you have gpu then run : 
```
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
Otherwise :  
```
pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Dataset
1. Download the dataset from [here](https://drive.google.com/file/d/1vS9Lhy1XOs-WnWJ-0MPYkFEImt03-B13/view?usp=sharing).
2. Also create your own dataset and maintain the following subdirectories and files formate : 
```
                hand_dataset  
                ├── test  
                │   └── different scenes  
                |       └── all frames  
                ├── train  
                |   └── all different scene  
                |       ├── annotations  
                |       |   └── all frames annotations  
                |       └── data  
                |           └── all frames  
                └── validation  
                    └── all different scene  
                        ├── annotations  
                        |   └── all frames annotations  
                        └── data  
                            └── all frames  
 ```
 
## Training
```
python3 train.py -d 'full_path_of_dataset_root' 
```
Run the command for training :
```
python3 train.py -d 'hand_dataset'  
```
By default dataset path is 'hand_dataset'.

## Evaluate
**Evaluate on a video by trained model**
```
python3 evaluation.py -m 'trained_model_path' -d 'frames_path_of_video' -a 'frame_annotations_path' -of 'reference_object_frame' -s save_the_output(True/False) -o 'output_video_path'
```
Run the command for evaluation :  
```
python3 evaluation.py -m 'experiment/best.pth.tar' -d 'hand_dataset/train/0001/data' -a 'hand_dataset/train/0001/annotations' -of '000000.jpeg' -s True -o 'evaluation.mp4'
```
**Evaluate on a video by opencv optical flow algorithm**    
```
python3 optical_flow_evaluation.py -d 'frames_of_video' -a 'frame_annotations_path' -of 'reference_object_frame' -s save_the_output(True/False) -o 'output_video_path'
```
Run the command for evaluation by optical flow :
```
python3 evaluation.py -d 'hand_dataset/train/0001/data' -a 'hand_dataset/train/0001/annotations' -of '000000.jpeg' -s True -o 'optical_flow_evaluation.mp4'
```

## Tracking
Tracking one gesture on a video.
```
python3 track.py -m 'trained_model_path' -d 'frames_path_of_video' -of 'reference_object_frame' -s save_the_output(True/False) -o 'output_video_path'
```
Run the command for Tracking :  
```
python3 track.py -m 'experiment/best.pth.tar' -d 'hand_dataset/train/0001/data' -of '000000.jpeg' -s True -o 'track.mp4'
```
Then draw a bounding box by mouse on reference object frame. Then press enter.

## Recognition
```
pyhton3 recognise.py -m 'trained_model_path' -d 'frames_path_of_video' -s save_the_output(True/False) -o 'output_video_path'
```
Run the command for Recognise :
```
pyhton3 recognise.py -m 'experiment/best.pth.tar' -d 'classification/vid2' -s True -o 'recognise.mp4'
```








