"""
Created on Feb 15 2024 08:24

@author: ISAC - pettirsch
"""

import numpy as np

from Pseudo_Label_Creation.Utils.matching_utils import match_dets
import time

import pdb



# Create class BackgroundFilter
class BackgroundFilter:
    # Create function __init__
    def __init__(self, window_size, iou_thesh, score_thresh, detector):
        # Set window_size and iou_thesh
        self.window_size = window_size
        self.iou_thesh = iou_thesh
        self.score_thresh = score_thresh

        # Set detector
        self.detector = detector

        # Storing parameters
        self.curr_background = None
        self.curr_low_thresh = None
        self.curr_high_thresh = None

        self.curr_detections_background = None
        self.curr_dets_background = None

    # Create function filter
    def filter(self, frames, frame_num, detections):

        start = time.time()

        det_idx_to_remove = []

        # Check if there are enough frames for background filter
        if len(frames) < self.window_size:
            return det_idx_to_remove

        # Compute background
        self.curr_background, use_cached_bg = self.compute_background(frames, frame_num)

        # Detect on Background
        if not use_cached_bg or self.curr_detections_background is None:
            self.curr_detections_background, self.curr_dets_background = self.detector.detect(self.curr_background)

        # Check if there are any background detections
        if len(self.curr_detections_background[0]["boxes"]) == 0:
            return det_idx_to_remove

        # Match Detections with Background
        matching_rows = match_dets(detections, self.curr_dets_background, iou_threshold=self.iou_thesh)
        if len(matching_rows) == 0:
            return det_idx_to_remove

        det_idx_to_remove = [x[0] for x in matching_rows if x[1] != -1 and detections[x[0],4] < self.score_thresh]

        return det_idx_to_remove

    def compute_background(self, frames, frame_num):
        use_cached_bg = False
        if len(frames) == self.window_size:
            low_thresh = 0
            high_thresh = self.window_size
        else:
            low_thresh = max(frame_num - int(self.window_size/2),0)
            high_thresh = low_thresh + self.window_size
            if high_thresh >= len(frames):
                high_thresh_offset = high_thresh - len(frames)
                high_thresh = len(frames)
                low_thresh = low_thresh - high_thresh_offset

        if self.curr_low_thresh is None:
            self.curr_low_thresh = low_thresh
            self.curr_high_thresh = high_thresh
        elif self.curr_low_thresh == low_thresh and self.curr_high_thresh == high_thresh:
            use_cached_bg = True
            return self.curr_background, use_cached_bg
        else:
            self.curr_low_thresh = low_thresh
            self.curr_high_thresh = high_thresh
        if frame_num + high_thresh > len(frames):
            high_thresh = len(frames) - frame_num
            low_thresh = self.window_size - high_thresh

        low_thresh = int(low_thresh)
        high_thresh = int(high_thresh)

        frames_for_background = frames[low_thresh:low_thresh + high_thresh]
        frames_for_background_np = np.asarray(frames_for_background, dtype=np.uint8)

        # Compute the pixel-wise mean
        background = np.mean(frames_for_background_np, axis=0).astype(np.uint8)

        return background, use_cached_bg

    def vanilla_background_calc(self, frames, frame_num):
        self.curr_background, _ = self.compute_background(frames, frame_num)

    def get_background(self):
        return self.curr_background

    def get_window_size(self):
        return self.window_size

    def clear_all(self):
        self.curr_background = None
        self.curr_low_thresh = None
        self.curr_high_thresh = None