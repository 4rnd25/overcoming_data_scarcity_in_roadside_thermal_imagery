"""
Created on Jan 12 10:48

@author: ISAC - pettirsch
"""

import os
import pdb

import cv2

from context_module.Input_Handling.inputHandler import InputHandler

# Create class Folderhandler
class Folderhandler(InputHandler):
    # Constructor
    def __init__(self, input_path):

        # Call super constructor
        super().__init__()

        # Check if input_path exists
        if not os.path.exists(input_path):
            # Raise error
            raise FileNotFoundError("Input path does not exist")

        # Set input_path
        self.input_path = input_path

        # Create list with all files in input_path if they are images
        self.file_list = []
        for file in os.listdir(self.input_path):
            if file.endswith(".png") or file.endswith(".jpg"):
                self.file_list.append(os.path.join(self.input_path, file))

        # If file_list is empty, raise error
        if len(self.file_list) == 0:
            # Raise error
            raise FileNotFoundError("No images found in input path")

        # Sort file_list by name
        self.file_list.sort()

    # Create function get_next_frame
    def _get_next_frame(self):
        # Check if finished
        if self.finished:
            # Raise error
            raise Exception("Input is finished")

        # Read image
        if self.frame_counter_internal >= len(self.file_list):
            return None
        else:
            frame = cv2.imread(self.file_list[self.frame_counter_internal])
            self.frame_counter_internal += 1

        # Check if frame is None
        if frame is None:
            # Return None
            return None

        # Return frame
        return frame

    def get_next_frame(self):

        if self.frame_counter >= len(self.file_list):
            self.finished = True
            return None, self.file_list[self.frame_counter-1]

        frame = self.get_frame_by_id(self.frame_counter, remove = False)
        self.frame_counter += 1

        next_frame_loading = self._get_next_frame()
        if next_frame_loading is not None:
            self.frames.append(next_frame_loading)
            self.frame_ids.append(self.frame_counter_internal)


        return frame, self.file_list[self.frame_counter-1]

    # Create function finished
    def is_finished(self):
        # Return finished
        return self.finished

    def is_video_folder_handler(self):
        return False

    def get_video_size(self):
        return len(self.file_list)

