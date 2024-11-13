"""
Created on Feb 15 2024 08:35

@author: ISAC - pettirsch
"""
import os

from Pseudo_Label_Creation.Input_Handling.folderhandler import Folderhandler
from Pseudo_Label_Creation.Input_Handling.video_file_handler import VideoFileHandler
from Pseudo_Label_Creation.Input_Handling.video_folder_handler import VideoFolderHandler


def input_loading(inputPath=None):

    assert os.path.exists(inputPath), "Input path does not exist"

    # Check if input path is a folder
    if os.path.isdir(inputPath):
        if len([name for name in os.listdir(inputPath) if name.endswith(".png") or name.endswith(".jpg")]) > 0:
            input_Handler = Folderhandler(inputPath)
        elif len([name for name in os.listdir(inputPath) if name.endswith(".mp4") or name.endswith(".avi") or name.endswith(
                ".mkv") or name.endswith(".mov")]) > 0:
            input_Handler = VideoFolderHandler(inputPath)
        else:
            # Raise input error if input path is neither a folder of images nor a folder of videos
            raise NotImplementedError("Input is neither a folder of images nor a folder of videos")
    # Check if input path is a video file and ends with .mp4, .avi, .mkv, .mov
    elif inputPath.endswith(".mp4") or inputPath.endswith(".avi") or inputPath.endswith(".mkv") or inputPath.endswith(
            ".mov"):
        input_Handler = VideoFileHandler(inputPath)
    else:
        # Raise not implemented error
        raise NotImplementedError("Input is neither a folder of images nor a folder of videos nor a video file")

    return input_Handler