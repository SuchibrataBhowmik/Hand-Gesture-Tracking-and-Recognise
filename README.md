# Hand Gesture Tracking and Recognition by Fully-Convolutional Siamese Network

## Introduction
This repository contains the codes and results to track hand gesture and recognise on my own dataset. This project is the Pytorch implementation of Fully-Convolutional Siamese Network.  
The project is divided into four major parts : **Training, Evaluation, Tracking, Recognise.**

## Environment setup
Clone this repository   https://github.com/SuchibrataBhowmik/Hand-Gesture-Tracking-and-Recognise.git  
This code is tested on 


## Dataset
1. Download the dataset from [here](https://drive.google.com/file/d/1vS9Lhy1XOs-WnWJ-0MPYkFEImt03-B13/view?usp=sharing)
2. Also create your own dataset and maintain the following subdirectories and files formate : 

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
                           
## Training
    python3 train.py -d full_path_of_dataset_root  
By default dataset path is 'hand_dataset'
    
    python3 train.py -d 'hand_dataset'  
    
## Evaluate
**Evaluate on a video by trained model**
  
    python3 evaluation.py
    
**Evaluate on a video by opencv optical flow algorithm**    

    python3 optical_flow_evaluation.py
    


















