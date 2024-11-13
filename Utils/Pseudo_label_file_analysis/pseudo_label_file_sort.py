"""
Created on Mar 13 2024 08:35

@author: ISAC - pettirsch
"""
import os
import argparse
import pdb
import math
import shutil
import xml.etree.ElementTree as ET


def sort_label_folder(label_file_path = None, raw_image_path = None, Video_folder = None, output_path = None,
                      num_train_per_cam = None, num_val_per_cam = None, student_label_file = None, teacher_label_file = None):

    video_list = []
    video_cam_match = []
    camera_list = ["BueLaMo_Cam_5", "BueLaMo_Cam_4", "Gewoehneffekte-02", "gewoehneffekte_Cam_4",
                   "gewoehneffekte_Cam_6", "Panoramastrasse_Cam_1", "Panoramastrasse_Cam_3", "13001"]
    numer_of_labels_per_Cam = [0, 0, 0, 0, 0, 0, 0, 0]
    videos_per_cam = [[], [], [], [], [], [], [], []]
    number_of_labels_per_video_per_cam = [[], [], [], [], [], [], [], []]
    label_list_per_video_per_cam = [[], [], [], [], [], [], [], []]
    cp_train_images_per_cam = [0 for i in range(len(camera_list))]
    cp_val_images_per_cam = [0 for i in range(len(camera_list))]

    # 1. Create video list and number of labels per video list and count number of videos per camera
    for video in os.listdir(Video_folder):
        print("video: ", video)
        if video.endswith(".mp4"):
            video_name = video.split(".mp4")[0]
            video_list.append(video_name)
        elif video.endswith(".mov"):
            video_name = video.split(".mov")[0]
            video_list.append(video_name)

        match = False
        for cam_name in camera_list:
            if cam_name in video:
                index = camera_list.index(cam_name)
                videos_per_cam[index].append(video_name)
                number_of_labels_per_video_per_cam[index].append(0)
                label_list_per_video_per_cam[index].append([])
                video_cam_match.append(index)
                match = True
        if not match:
            videos_per_cam[-1].append(video_name)
            number_of_labels_per_video_per_cam[-1].append(0)
            label_list_per_video_per_cam[-1].append([])
            video_cam_match.append(len(camera_list) - 1)

    # 2. Count numper of labels per video and camera
    for file in os.listdir(label_file_path):
        if file.endswith(".xml"):

            # Parse the XML file
            xml_file_path = os.path.join(label_file_path, file)
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            # Check if there are any objects in the XML file
            if root.find('object') is None:
                continue  # Skip this file if no <object> tag is found

            for video_name in video_list:
                if video_name in file:
                    index_video_all = video_list.index(video_name)
                    cam_index = video_cam_match[index_video_all]
                    index_video_cam = videos_per_cam[cam_index].index(video_name)
                    number_of_labels_per_video_per_cam[cam_index][index_video_cam] += 1
                    numer_of_labels_per_Cam[cam_index] += 1
                    label_list_per_video_per_cam[cam_index][index_video_cam].append(os.path.join(label_file_path, file))

    # 3. Create output folder
    output_path = os.path.join(output_path, "Pseudo_Labels_sorted")
    os.makedirs(output_path, exist_ok=True)
    train_output = os.path.join(output_path, "train")
    val_output = os.path.join(output_path, "val")
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(val_output, exist_ok=True)

    # Iterate over all cameras
    for i in range(len(camera_list)):
        num_copied_labels = 0

        # Sort number_of_labels_per_video_per_cam[i] and label_list_per_video_per_cam[i] in the same order
        num_of_labels = number_of_labels_per_video_per_cam[i]
        label_list_per_cam = label_list_per_video_per_cam[i]
        video_cam = videos_per_cam[i]
        label_list_per_cam = [x for _, x in sorted(zip(num_of_labels, label_list_per_cam))]
        videos_cam = [x for _, x in sorted(zip(num_of_labels, video_cam))]
        num_of_labels.sort()
        print(f"Camera {camera_list[i]}: {num_of_labels}")

        # Calc optimal number of labels per video
        print("len(videos_per_cam[i]): ", len(videos_per_cam[i]))
        opt_num_labels = math.floor((num_train_per_cam + num_val_per_cam) / len(videos_per_cam[i]))
        print(f"Optimal number of labels per video: {opt_num_labels}")

        # Iterate over all videos of the camera
        for j in range(len(videos_per_cam[i])):

            opt_num_labels = math.floor((num_train_per_cam + num_val_per_cam - num_copied_labels) / (len(videos_per_cam[i])-j))
            print(f"Optimal number of labels per video: {opt_num_labels}")

            if j == len(videos_per_cam[i]) - 1:
                print("num copied labels: ", num_copied_labels)
                opt_num_labels = (num_train_per_cam + num_val_per_cam) - num_copied_labels

            num_labels = num_of_labels[j]
            label_list = label_list_per_cam[j]
            label_list.sort()

            # Check if video is only for train or also for val or only for val
            if num_copied_labels+min(num_labels, opt_num_labels) <= num_train_per_cam:
                output_folder = [os.path.join(train_output, camera_list[i])]
            elif num_copied_labels > num_train_per_cam:
                output_folder = [os.path.join(val_output, camera_list[i])]
            else:
                output_folder = [os.path.join(train_output, camera_list[i]),
                    os.path.join(val_output, camera_list[i])]
            for spez_output_folder in output_folder:
                os.makedirs(spez_output_folder, exist_ok=True)

            # Copy labels to output folder
            if len(output_folder) == 1:
                for file in label_list[:min(num_labels, opt_num_labels)]:
                    shutil.copy(file, output_folder[0])
                    file_new = file.replace(label_file_path, raw_image_path)
                    file_new = file_new.replace(".xml", ".png")
                    shutil.copy(file_new, output_folder[0])
                    num_copied_labels += 1
            else:
                num_missing_train_labels = num_train_per_cam - num_copied_labels
                num_missing_val_labels = min(num_labels, opt_num_labels) - num_missing_train_labels
                for file in label_list[:num_missing_train_labels]:
                    shutil.copy(file, output_folder[0])
                    file_new = file.replace(label_file_path, raw_image_path)
                    file_new = file_new.replace(".xml", ".png")
                    shutil.copy(file_new, output_folder[0])
                    num_copied_labels += 1
                for file in label_list[num_missing_train_labels:num_missing_train_labels+num_missing_val_labels]:
                    shutil.copy(file, output_folder[1])
                    file_new = file.replace(label_file_path, raw_image_path)
                    file_new = file_new.replace(".xml", ".png")
                    shutil.copy(file_new, output_folder[1])
                    num_copied_labels += 1

            # Recalculate opt_num_labels
            if min(num_labels, opt_num_labels) < opt_num_labels:
                if j == len(videos_per_cam[i]) - 1:
                    assert 2==1
                offset = (opt_num_labels - min(num_labels, opt_num_labels)) / (len(videos_per_cam[i]) - j)
                opt_num_labels += offset
                opt_num_labels = math.ceil(opt_num_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_file_path", help="Path to label file folder", default="")
    parser.add_argument("--raw_image_path", help="Path to raw image file folder", default="")
    parser.add_argument("--Video_folder", help="Path with all videos", default="")
    parser.add_argument("--output_path", help="Path to output folder", default="")
    parser.add_argument("--num_train_per_cam", help="Number of training images per camera", default=600)
    parser.add_argument("--num_val_per_cam", help="Number of validation images per camera", default=200)
    args = parser.parse_args()

    sort_label_folder(args.label_file_path, args.raw_image_path, args.Video_folder, args.output_path,
                      int(args.num_train_per_cam), int(args.num_val_per_cam))