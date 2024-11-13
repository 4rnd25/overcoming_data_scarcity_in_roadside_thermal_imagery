"""
Created on March 19 2024 09:33

@author: ISAC - pettirsch
"""
import os
import pdb

import numpy as np
from matplotlib import pyplot as plt
import json
import argparse


def evaluate_yolov7_datasets(train_folder=None, val_Folder=None, test_Folder_1=None, test_Folder_2=None,
                             class_name_to_id_mapping={"Motorcycle": 0, "Car": 1, "Truck": 2, "Bus": 3, "Person": 4,
                                                       "Bicycle": 5, "E-Scooter": 6}, image_size=(640, 480),
                             threshold_hard=128, threshold_easy=384, output_path=None,
                             data_splits=["X-t_A_Train", "X-t_A_Val", "X-t_A_Test", "X-t_B1_Test"],
                             dataset_colors=[(85 / 255, 142 / 255, 213 / 255), (142 / 255, 180 / 255, 227 / 255),
                                             (198 / 255, 217 / 255, 241 / 255), (237 / 255, 206 / 255, 204 / 255)]):
    """
    Evaluate the YOLOv7 datasets by counting the number of images and labels in each split.

    :param train_folder: Path to the training folder
    :param val_Folder: Path to the validation folder
    :param test_Folder_1: Path to the first test folder
    :param test_Folder_2: Path to the second test folder
    :param class_name_to_id_mapping: Dictionary to map class names to ids
    :return: None
    """

    # Create dict to store results:
    data_distribution_per_class = {}
    for key in class_name_to_id_mapping.keys():
        data_distribution_per_class[key] = {}
        for split in data_splits:
            data_distribution_per_class[key][split] = {"hard": 0, "medium": 0, "easy": 0}

    data_split_sums = [0 for split in data_splits]

    # Create dict to map class ids to class names
    class_id_to_name_mapping = {v: k for k, v in class_name_to_id_mapping.items()}

    # Create list with all folders
    folder_list = [train_folder, val_Folder, test_Folder_1, test_Folder_2]
    folder_name_list = data_splits

    # For each folder
    for folder, folder_name in zip(folder_list, folder_name_list):
        print("Processing folder: {}".format(folder_name))
        # Iterate over all files in the folder
        for file in os.listdir(folder):
            if file.endswith(".txt"):
                # Open the file
                with open(os.path.join(folder, file), "r") as file:
                    # Read the file
                    annotation_list = file.read().split("\n")[:-1]
                    annotation_list = [x.split(" ") for x in annotation_list]
                    annotation_list = [[float(y) for y in x] for x in annotation_list]

                    # Transform the annotations
                    annotations = np.array(annotation_list)
                    w, h = image_size

                    transformed_annotations = np.copy(annotations)
                    transformed_annotations[:, [1, 3]] = annotations[:, [1, 3]] * w
                    transformed_annotations[:, [2, 4]] = annotations[:, [2, 4]] * h

                    transformed_annotations[:, 1] = transformed_annotations[:, 1] - (
                            transformed_annotations[:, 3] / 2)
                    transformed_annotations[:, 2] = transformed_annotations[:, 2] - (
                            transformed_annotations[:, 4] / 2)
                    transformed_annotations[:, 3] = transformed_annotations[:, 1] + transformed_annotations[:, 3]
                    transformed_annotations[:, 4] = transformed_annotations[:, 2] + transformed_annotations[:, 4]

                    # Analyse each annotation
                    for ann in transformed_annotations:
                        obj_cls, x0, y0, x1, y1 = ann

                        # Calc object area
                        area = (x1 - x0) * (y1 - y0)
                        if area < threshold_hard:
                            difficulty = "hard"
                        elif area > threshold_easy:
                            difficulty = "easy"
                        else:
                            difficulty = "medium"

                        # Get the class name
                        class_name = class_id_to_name_mapping[(int(obj_cls))]

                        # Update the dict
                        data_distribution_per_class[class_name][folder_name][difficulty] += 1

                        # Update the sum
                        data_split_sums[data_splits.index(folder_name)] += 1

    # Save dict as json
    with open(os.path.join(output_path, "dataset_analysis.json"), "w") as file:
        json.dump(data_distribution_per_class, file)

    # Plot the results
    # hatches = {"easy": "/", "medium": "\\", "hard": "*"}
    alphas = {"easy": 1, "medium": 0.75, "hard": 0.5}

    # Plot the results
    plt.figure(figsize=(20, 14))
    plt.rcParams.update({'font.size': 24, 'font.family': 'Times New Roman'})

    # Extracting class names
    classes = list(data_distribution_per_class.keys())

    # Number of classes
    num_classes = len(classes)

    # Bar width
    cls_bar_width = 0.8
    bar_width = cls_bar_width / 4

    # X positions for the bars
    x = np.arange(num_classes)
    x = x + 1

    # Plotting the bars
    for i, (key, val) in enumerate(data_distribution_per_class.items()):
        x_pos_cls = x[i]
        height_bar_save = []
        for j, (dataset, dataset_values) in enumerate(val.items()):
            x_pos_dataset = x_pos_cls + (j - 1.5) * bar_width
            values = [dataset_values["hard"], dataset_values["medium"], dataset_values["easy"]]
            total_value = sum(values)
            height_bar = total_value # / data_split_sums[j]

            heights_diff = [(dataset_values["easy"]),
                            (dataset_values["medium"]),
                            (dataset_values["hard"])]
            #heights_diff = [round(x, 2) * 100 for x in heights_diff]
            diffs = ["easy", "medium", "hard"]

            for q, height in enumerate(heights_diff):
                if q == 0:
                    bottom = 0
                else:
                    bottom = np.sum(heights_diff[:q], axis=0)
                print(height)
                plt.bar(x=x_pos_dataset, height=height, width=bar_width,
                        label=dataset + " " + diffs[q] if i == 0 else None, color=dataset_colors[j],
                        alpha=alphas[diffs[q]],
                        bottom=bottom, edgecolor='black', linewidth=1.5)
            height_bar_save.append(height_bar)

        max_height = max(height_bar_save)
        for w, height_bar in enumerate(height_bar_save):
            print(max_height)
            x_pos_dataset = x_pos_cls + (w - 1.5) * bar_width
            plt.text(x_pos_dataset, (round(max_height, 2)) + 500, str(round(height_bar, 2)), #+ "%",
                     ha='center', color='black',
                     rotation=90)

    # Set x-axis labels and ticks
    plt.xticks(x, classes)

    # Add legend and labels
    plt.xlabel('Classes')
    plt.ylabel('Absolute number per Dataset')
    plt.ylim(0, 25000)
    plt.title('Dataset Analysis')
    plt.legend()

    # Save the plot
    # plt.tight_layout()
    plt.savefig(os.path.join(output_path, "dataset_analysis.png"))


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Evaluate YOLOv7 datasets")
    parser.add_argument("--train_folder", type=str, help="Path to the training folder")
    parser.add_argument("--val_Folder", type=str, help="Path to the validation folder")
    parser.add_argument("--test_Folder_1", type=str, help="Path to the first test folder")
    parser.add_argument("--test_Folder_2", type=str, help="Path to the second test folder")
    parser.add_argument("--output_path", type=str, help="Path to the output folder")
    args = parser.parse_args()

    #data_splits = ["X-t_A_Train", "X-t_A_Val", "X-t_A_Test", "X-t_B1_Test"]
    data_splits = ["X-B1_Train_Val_V1", "X-B1_Train_Val_V2", "X-t_A_Test", "X-t_B1_Test"]
    # dataset_colors = [(85 / 255, 142 / 255, 213 / 255), (142 / 255, 180 / 255, 227 / 255),
    #                   (198 / 255, 217 / 255, 241 / 255), (237 / 255, 206 / 255, 204 / 255)]
    dataset_colors = [(149 / 255, 55 / 255, 53 / 255), (217 / 255, 150 / 255, 148 / 255),
                      (198 / 255, 217 / 255, 241 / 255), (237 / 255, 206 / 255, 204 / 255)]


    # Evaluate the datasets
    evaluate_yolov7_datasets(train_folder=args.train_folder, val_Folder=args.val_Folder,
                             test_Folder_1=args.test_Folder_1,
                             test_Folder_2=args.test_Folder_2, output_path=args.output_path,
                             data_splits=data_splits,
                             dataset_colors=dataset_colors)
