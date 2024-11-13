"""
Created on Feb 15 2024 08:43

@author: ISAC - pettirsch
"""
import os

import numpy as np

from context_module.Input_Handling.video_file_handler import VideoFileHandler
from context_module.Input_Handling.inputHandler import InputHandler

# Create class VideoFolderHandler
class VideoFolderHandler(InputHandler):
    # Constructor
    def __init__(self, input_path):

        # Call super constructor
        super().__init__()

        # Check if input_path exists
        if not os.path.exists(input_path):
            # Raise error
            raise FileNotFoundError("Input path does not exist")

        # Get all video files in input_path
        self.video_files = [os.path.join(input_path,f) for f in os.listdir(input_path) if
                       f.endswith(".mp4") or f.endswith(".avi") or f.endswith(".mkv") or f.endswith(".mov")]
        self.video_idx = 0

        # Check if video_files is empty
        if len(self.video_files) == 0:
            # Raise error
            raise FileNotFoundError("No video files found in input path")

        # Set self.videoHandler to None
        self.videoHandler = None

        self.cap_counter = 0


    # Create function get_next_frame
    def _get_next_frame(self):
        # Check if finished
        if self.finished:
            # Raise error
            raise Exception("Input is finished")

        if self.videoHandler is not None:
            if len(self.videoHandler.get_frame_ids()) == 0:
                self.videoHandler = None

        # Check if videoHandler is None
        if self.videoHandler is None:
            if self.video_idx == len(self.video_files):
                # Set finished to true
                self.finished = True
                # Return None
                return None, None
            else:
                if self.videoHandler is not None:
                    self.videoHandler.clear_all()
                    self.videoHandler = None
                # Create VideoFileHandler
                self.videoHandler = VideoFileHandler(self.video_files[self.video_idx])
                self.n_frames = self.videoHandler.get_video_size()
                self.videoHandler.load_first_frames(self.n_frames-1)
                self.max_frames = self.videoHandler.get_max_frames()
                self.video_idx += 1
                self.frame_counter = 0

        # Get next frame from videoHandler
        frame = self.videoHandler._get_next_frame()

        # Check if videoHandler is finished
        if self.videoHandler.is_finished():
            # Set videoHandler to None
            self.videoHandler = None
            self.frames = np.asarray([])

        # Check if finished
        if self.video_idx == len(self.video_files) and self.videoHandler is None:
            # Set finished to true
            self.finished = True

        # Return frame
        return frame

    def get_next_frame(self):

        if self.videoHandler is None:
            if self.video_idx == len(self.video_files):
                print("self.video_idx and len(self.video_files): ", self.video_idx, len(self.video_files))
                self.finished = True
                return None, None
            else:
                print("Set vidoeHandler, n_frames = ", self.n_frames)
                self.frames = np.asarray([])
                if self.videoHandler is not None:
                    self.videoHandler.clear_all()
                    self.videoHandler = None
                self.videoHandler = VideoFileHandler(self.video_files[self.video_idx])
                self.videoHandler.load_first_frames(self.n_frames-1)
                self.max_frames = self.videoHandler.get_max_frames()
                self.video_idx += 1

        frame, path = self.videoHandler.get_next_frame()
        self.cap_counter= self.videoHandler.frame_counter_internal

        self.frames = self.videoHandler.get_frames()
        self.frame_ids = self.videoHandler.get_frame_ids()
        self.frame_counter = self.videoHandler.get_frame_num()

        # Check if videoHandler is finished
        if self.videoHandler.is_finished():
            # Set videoHandler to None
            self.max_frames = self.videoHandler.get_max_frames()
            self.videoHandler = None

        return frame, path


    # Create function finished
    def is_finished(self):
        # Return finished
        return self.finished

    def is_video_folder_handler(self):
        return True

    def getVideoIdx(self):
        return self.video_idx

    def set_first_frames(self, n_frames):
        self.n_frames = n_frames

    def get_video_size(self):
        if self.videoHandler is None:
            if self.video_idx == len(self.video_files):
                # Set finished to true
                self.finished = True
                # Return None
                return None
            else:
                # Create VideoFileHandler
                self.videoHandler = VideoFileHandler(self.video_files[self.video_idx])
                self.n_frames = self.videoHandler.get_video_size()
                self.videoHandler.load_first_frames(self.n_frames-1)
                self.max_frames = self.videoHandler.get_max_frames()
                self.video_idx += 1
                self.frame_counter = 0

                return self.videoHandler.get_video_size()

    def clear_frames(self):
        self.frames = []
        if self.videoHandler is not None:
            self.videoHandler.clear_frames()

    def _get_max_frames(self):
        if self.videoHandler is not None:
            return self.videoHandler.get_max_frames()
        else:
            return self.max_frames