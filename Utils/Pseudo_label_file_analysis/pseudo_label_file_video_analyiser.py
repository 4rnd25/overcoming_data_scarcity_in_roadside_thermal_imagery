"""
Created on Mar 13 2024 08:35

@author: ISAC - pettirsch
"""
import csv
import os
import argparse

def anaylse_label_folder(folderPath = None, videoFolder = None, outputPath = None, newVideoFolder = None):

    video_list = []
    numer_of_labels_per_Video = []
    camera_list = ["BueLaMo_Cam_5", "BueLaMo_Cam_4", "Gewoehneffekte-02", "gewoehneffekte_Cam_4",
                   "gewoehneffekte_Cam_6", "Panoramastrasse_Cam_1", "Panoramastrasse_Cam_3", "13001"]
    numer_of_labels_per_Cam = [0,0,0,0,0,0,0,0]

    for video in os.listdir(videoFolder):
        if video.endswith(".mp4"):
            video_name = video.split(".mp4")[0]
            video_list.append(video_name)
            numer_of_labels_per_Video.append(0)
        elif video.endswith(".mov"):
            video_name = video.split(".mov")[0]
            video_list.append(video_name)
            numer_of_labels_per_Video.append(0)

    for file in os.listdir(folderPath):
        if file.endswith(".xml"):
            for video_name in video_list:
                print(video_name)
                if video_name in file:
                    index = video_list.index(video_name)
                    numer_of_labels_per_Video[index] += 1

            match = False
            for cam_name in camera_list:
                if cam_name in file:
                    index = camera_list.index(cam_name)
                    numer_of_labels_per_Cam[index] += 1
                    match = True
            if not match:
                numer_of_labels_per_Cam[-1] += 1

    # Get only folder_name of outputpath
    folder_name = os.path.basename(outputPath)
    folder_name_cam = folder_name + "_cam"

    # Save as csv where video_list is column 0 and number of labels is column 2 with headings "video" and "number of labels"
    csv_path = os.path.join(outputPath, folder_name + '_analysis.csv')
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['video', 'number of labels'])
        for video, label_count in zip(video_list, numer_of_labels_per_Video):
            writer.writerow([video, label_count])

    csv_path = os.path.join(outputPath, folder_name_cam + '_analysis.csv')
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Camera', 'number of labels'])
        for cam, label_count in zip(camera_list, numer_of_labels_per_Cam):
            writer.writerow([cam, label_count])

    if newVideoFolder is not None:
        for video_filename in os.listdir(videoFolder):
            if video_filename.endswith(".mp4"):
                video_filename_base = video_filename.split(".mp4")[0]
            if video_filename.endswith(".mov"):
                video_filename_base = video_filename.split(".mov")[0]

            video_idx = video_list.index(video_filename_base)
            num_labels = numer_of_labels_per_Video[video_idx]
            if num_labels > 1:
                print(f"More than one label found for video {video_filename_base}")
            if num_labels == 0:
                print(f"No labels found for video {video_filename_base}")
                video_path = os.path.join(videoFolder, video_filename)
                new_video_path = os.path.join(newVideoFolder, video_filename)
                os.system(f"cp {video_path} {new_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", help="Path to label file folder", default=True)
    parser.add_argument("--Video_folder", help="Path with all videos", default=True)
    parser.add_argument("--output_path", help="Path to output file", default=True)
    parser.add_argument("--new_Video_folder", help="Path to new video folder", default=None)
    args = parser.parse_args()

    anaylse_label_folder(args.file_path, args.Video_folder, args.output_path, args.new_Video_folder)

