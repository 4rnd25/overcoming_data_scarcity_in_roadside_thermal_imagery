"""
Created on Feb 15 2024 10:09

@author: ISAC - pettirsch
"""
import numpy as np


class InputHandler:
    def __init__(self):
        self.frames = []
        self.frame_ids = []
        self.frame_counter = 0
        self.frame_counter_internal = 0
        self.finished = False
        self.n_frames = 0
        self.max_frames = 0

    def load_first_frames(self, n_frames):
        self.n_frames = int(n_frames)
        n_frames = int(n_frames)
        for _ in range(n_frames):
            frame = self._get_next_frame()
            if frame is None:
                break
            self.frames.append(frame)
            self.frame_ids.append(self.frame_counter_internal-1)
        self.max_frames = self._get_max_frames()
        self.frame_counter= 0

    def _get_next_frame(self):
        raise NotImplementedError("Subclasses must implement get_next_frame() method")

    def get_frames(self):
        return self.frames

    def get_max_frames(self):
        self.max_frames = self._get_max_frames()
        return self.max_frames

    def get_frame_num(self):
        return self.frame_counter

    def clear_frames(self):
        self.frames = []

    def clear_all(self):
        self.frames = []
        self.frame_counter = 0
        self.finished = False
        self.n_frames = 0

    def get_frame_by_id(self, frame_id, remove = True):
        try:
            if frame_id > self.frame_ids[-1] or frame_id < self.frame_ids[0]:
                return None
            idx = self.frame_ids.index(frame_id)
            frame = self.frames[idx]
            if remove:
                self.frames.pop(idx)
                self.frame_ids.pop(idx)
        except:
            print("Frame_id: ", frame_id)
            print("self.frame_ids: ", self.frame_ids)

        return frame

    def get_frame_ids(self):
        return self.frame_ids