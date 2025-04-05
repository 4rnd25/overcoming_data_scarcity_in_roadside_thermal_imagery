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
7. [Acknowledgements](#acknowledgements)
8. [License](#license)

## Introduction 
This is the official repository for the paper: Overcoming data scarcity in roadside thermal imagery: A new dataset and weakly supervised incremental learning framework. As reference to this code please cite the paper. 

## Installation
This code was build using python 3.11. The requirements.txt file includes all necessary packages. Additionally this repo should be part of the PYTHONPATH. Moreover it is important to clone with --recurse-submodules to get also the modified yolov7 and the SORT code.


## Data
The folder datahandling contains all necessary files: <br>
 
	- yolov7_dataset_creator.py transfers the images and .csv files to yolov7 format <br> 
	- aau_to_yolo.py does the same fo AAU dataset <br>
	- adapt to automotive thermal for the AT dataset with additional class mapping by automotive_thermal_cls_map.py <br>
	- yolov7_dataset_evaluator.py enables to plot the class distribution  <br>
	- yolov7_dataset_tester.py plots random images and annotations from the dataset to visible check them 

## Contextmodule
All important settings are made in the Configs. All configs used in the paper could be found there. It is important to clone the submodule context_module/Tracking/SORT as it is used for tracking (please consider the license file). Pseudo-Label creation is started using the file Main-pseudo_label_creation.py for single config or pseudo_label_creation_configRunner.py for multiple configs. 

## Training
- For the general domain adaption and training on xtA the original train.py file could be used.
- For training with the teacher network and the context module: train_with_teacher_and_context_module.py
- For training with the teacher network, the context module and the remember module: train_with_teacher_context_module_and_remember_module.py
- For training with the teacher network, the context module and the remember module with previous images in val: train_with_teacher_context_module_and_remember_module_mixed_val.py
- During all experiments with yolo-tiny:  yolov7/cfg/training/yolov7-tiny_cls_7_IDetect.yaml  and yolov7/data/hyp.scratch.tiny_therm_no_aug.yaml were used with batch size 16. 
- During training of the teacher: yolov7/cfg/training/yolov7_cls_7_IDetect.yaml and yolov7/data/hyp.scratch.costum_therm_no_aug.yaml were used with batch size 16.

## Evaluation
- Models can be evaluated using Evaluation/model_evaluator.py
- Pseudo labels can be evaluated using Evaluation/pseudo_label_evaluator.py or for multiple pseudo-labels: Evaluation/pseudo_label_folder_evaluator.py
- The inference time could be mesured with yolov7/detect.py and for the onnx models with Evaluation/inference_time.py

## Acknowledgements
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
  Changed files: -
  Added files: train_with_teacher_and_context_module.py, train_with_teacher_context_module_and_remember_module.py, train_with_teacher_context_module_and_remember_module_mixed_val.py, test_with_remember_module.py, yolov7/utils/loss.py yolov7/utils/general.py yolov7/utils/datasets.py, yolov7/data/hyp.scratch.tiny_therm_no_aug.yaml, yolov7/cfg/training/yolov7_cls_7_IDetect.yaml, yolov7/data/hyp.scratch.tiny_therm_no_aug.yaml, yolov7/data/hyp.scratch.costum_therm_no_aug.yaml

## License
This repositroy is released under the GPL License (refer to the LICENSE file for details). 
