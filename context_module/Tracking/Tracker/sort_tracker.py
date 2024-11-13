"""
Created on Feb 15 2024 11:23

@author: ISAC - pettirsch
"""

from context_module.Tracking.SORT.sort import *
from context_module.Tracking.Tracker.tracker import Tracker
from context_module.Utils.matching_utils import match_dets

import pdb


class SortTracker(Tracker):
    def __init__(self, max_age=5, iou_threshold=0.3, min_hits=0):
        # Call parent constructor
        super().__init__()

        self.removed_tracks = {}
        self.sort_tracker = Sort(max_age=max_age, iou_threshold=iou_threshold, min_hits=min_hits)

        # Save parameters
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits

    def track(self, detections, dets, frame_num):

        # Update tracker
        curr_tracks = self.sort_tracker.update(dets)

        # Match detections with tracks
        matching_rows = match_dets(dets,curr_tracks)

        dets_without_track = [x for x in range(len(detections["boxes"]))]
        track_idx_list = []
        track_idx_list_intern = []
        for det_idx, track_idx in matching_rows:
            track_idx_intern = curr_tracks[track_idx,4]
            if str(track_idx_intern) in self.removed_tracks.keys():
                finsih_search = False
                while not finsih_search:
                    if str(track_idx_intern) in self.removed_tracks.keys():
                        track_idx_intern = self.removed_tracks[str(track_idx_intern)]
                    else:
                        finsih_search = True
            else:
                if track_idx_intern >= self.max_track_idx:
                    self.max_track_idx = track_idx_intern+1
            track_idx_list.append(track_idx_intern)
            track_idx_list_intern.append(track_idx)
            dets_without_track.remove(det_idx)
            self.tracked_objects[str(track_idx_intern)] = frame_num

        self.check_remove(frame_num)

        return track_idx_list, dets_without_track, curr_tracks, track_idx_list_intern

    def check_remove(self, frame_num):
        obj_to_pop = []
        for track_idx in self.tracked_objects.keys():
            if self.tracked_objects[track_idx] < frame_num - self.max_age:
                self.removed_tracks[track_idx] = self.max_track_idx
                obj_to_pop.append(track_idx)
                self.max_track_idx += 1

        for obj in obj_to_pop:
            self.tracked_objects.pop(obj)

    def clear_all(self):
        self.tracked_objects = {}
        self.max_track_idx = 0
        self.sort_tracker = None
        self.sort_tracker = Sort(max_age=self.max_age, iou_threshold=self.iou_threshold, min_hits=self.min_hits)
        self.removed_tracks = {}

