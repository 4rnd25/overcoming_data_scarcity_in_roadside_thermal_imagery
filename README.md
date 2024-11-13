# Overcoming data scarcity in roadside thermal imagery

Repository for paper: Overcoming data scarcity in roadside thermal imagery: A new dataset and weakly supervised incremental learning framework. <br>

Link to paper: 

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Data](#data)
4. [ContextModule](#contextmodule)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction 
This is the official repository for the paper: Overcoming data scarcity in roadside thermal imagery: A new dataset and weakly supervised incremental learning framework. As reference to this code please cite the paper. 

## Installation
This code was build using python 3.11. The requirements.txt file includes all necessary packages. Additionally this repo should be part of the PYTHONPATH.


## Data
The folder datahandling contains all necessary files: <br>
 
	- yolov7_dataset_creator.py transfers the images and .csv files to yolov7 format <br> 
	- aau_to_yolo.py does the same fo AAU dataset <br>
	- adapt to automotive thermal for the AT dataset with additional class mapping by automotive_thermal_cls_map.py <br>
	- yolov7_dataset_evaluator.py enables to plot the class distribution  <br>
	- yolov7_dataset_tester.py plots random images and annotations from the dataset to visible check them 

## Contextmodule
All important settings are made in the Configs. All configs used in the paper could be found there. It is important to clone the submodule context_module/Tracking/SORT as it is used for tracking (please consider the license file). Pseudo-Label creation is started using the file Main-pseudo_label_creation.py for single config or pseudo_label_creation_configRunner.py for multiple configs. 

## Contribution
This repositroy includes the following third-party submodules: <br>

- SORT located at context_module/Tracking/SORT - 
  Repository:https://github.com/abewley/sort - 
  License: GPL3.0 - 
  License details are available at: context_module/Tracking/SORT/LICENSE - 
  No files were changed

- yolov7 located at /yolov7 - 
  Repository: https://github.com/WongKinYiu/yolov7 - 
  License: GPL3.0 - 
  License details are available at: yolov7/LICENSE.md - 
  Changed files: 
