"""
Created on Feb 05 2024 10:52

@author: ISAC - pettirsch
"""

import os
import cv2
import pdb

from context_module.Input_Handling.inputHandler import InputHandler

# Create class VideoFileHandler
class VideoFileHandler(InputHandler):
    # Constructor
    def __init__(self, input_path):

        # Call super constructor
        super().__init__()

        # Check if input_path exists
        if not os.path.exists(input_path):
            # Raise error
            raise FileNotFoundError("Input path does not exist")

        # Create cv video capture
        self.cap = cv2.VideoCapture(input_path)

        # Check if camera opened successfully
        if (self.cap.isOpened() == False):
            # Raise Error
            raise Exception("Error opening video stream or file {}".format(input_path))
        else:
            print("Video file {} opened successfully".format(input_path))

        self.load_all_frames = False

        # Set input_path
        self.input_path = input_path

    # Create function get_next_frame
    def _get_next_frame(self):
        # Check if finished
        if self.finished:
            # Raise error
            raise Exception("Input is finished")

        # Read image
        ret, frame = self.cap.read()

        # Check if frame is None
        if frame is None:
            self.load_all_frames = True
            self.max_frames = self.frame_counter_internal-1
            # Return None
            return None
        else:
            self.frame_counter_internal += 1

        # Return frame
        return frame

    def get_next_frame(self):

        if self.frame_counter > self.get_max_frames():
            self.finished = True
            return None, self.input_path

        frame = self.get_frame_by_id(self.frame_counter, remove=False)
        if ".mov" in self.input_path:
            img_path = self.input_path.removesuffix(".mov") + "_" + str(self.frame_counter) + ".png"
        else:
            img_path = self.input_path.removesuffix(".mp4") + "_" + str(self.frame_counter) + ".png"
        self.frame_counter += 1

        next_frame_loading = self._get_next_frame()
        if next_frame_loading is not None:
            self.frames.append(next_frame_loading)
            self.frame_ids.append(self.frame_counter_internal-1)

        return frame, img_path

    # Create function finished
    def is_finished(self):
        # Return finished
        return self.finished

    def is_video_folder_handler(self):
        return False

    def get_video_size(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT)-1

    def _get_max_frames(self):
        if not self.load_all_frames:
            return self.get_video_size()
        else:
            return self.max_frames



