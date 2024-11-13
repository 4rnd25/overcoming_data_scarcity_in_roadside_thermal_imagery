"""
Created on Feb 15 2024 09:16

@author: ISAC - pettirsch
"""
import pdb

import numpy as np

from context_module.Filtering.Detection_Filter.background_filter import BackgroundFilter
from context_module.Filtering.Detection_Filter.pbde_filter import PBDEFilter
from context_module.Filtering.Detection_Filter.srrs_filter import SRRSFilter


# Create class detection_filter
class DetectionFilter:
    # Create function __init__
    def __init__(self, background_filter=False, background_filter_config=None, srrs_filter=False,
                 srrs_filter_config=None, pbde_filter=False, pbde_filter_config=None, detector=None,
                 save_filtered_objects=False, save_backgrounds=False):

        self.save_filtered_objects = save_filtered_objects
        self.filtered_objects = {}
        self.save_backgrounds = save_backgrounds
        self.backgrounds = {}

        # Check if background_filter is True
        if background_filter:
            assert background_filter_config is not None, "Background filter config is None"
            assert detector is not None, "Detector is None"
            # Create background_filter
            if "score_thresh" in background_filter_config:
                score_thresh = background_filter_config["score_thresh"]
            else:
                score_thresh = 0
            self.background_filter = BackgroundFilter(window_size=background_filter_config["window_size"],
                                                      iou_thesh=background_filter_config["iou_thresh"],
                                                      score_thresh=score_thresh,
                                                      detector=detector)
        else:
            self.background_filter = None
        self.background_filter_remove_idx = []

        # Check if srrs_filter is True
        if srrs_filter:
            assert srrs_filter_config is not None, "SRRS filter config is None"
            # Create srrs_filter
            self.srrs_filter = SRRSFilter(iou_thresh=srrs_filter_config["iou_thresh"],
                                          srrs_thresh=srrs_filter_config["srrs_thresh"])
            detector.set_get_raw_detections(True)
        else:
            self.srrs_filter = None
        self.srrs_filter_remove_idx = []

        # Check if pbde_filter is True
        if pbde_filter:
            assert pbde_filter_config is not None, "PBDE filter config is None"
            # Create pbde_filter
            detector.set_get_raw_detections(True)
            self.pbde_filter = PBDEFilter(pbde_filter_config["pbde_thresh"])
        else:
            self.pbde_filter = None
        self.pbde_filter_remove_idx = []

        self.detector = detector

    # Create function filter
    def filter(self, detections, dets, frame_num, frames):

        #print("detection filter frame_num: ", frame_num)
        #print("num detections: ", len(detections[0]["boxes"]))

        # Todo change to multi image processing
        detections = detections[0]
        if len(detections["boxes"]) == 0:
            self.filtered_objects[str(frame_num)] = []
            if self.background_filter is not None:
                self.background_filter.vanilla_background_calc(frames, frame_num)
                self.backgrounds[str(frame_num)] = self.background_filter.curr_background

            return detections, dets

        assert len(frames) > 0, "Frames is empty"

        # Check if background_filter exists
        if self.background_filter is not None:
            assert len(frames) >= self.background_filter.get_window_size(), "Not enough frames for background filter"

            self.background_filter_remove_idx = self.background_filter.filter(frames, frame_num, dets)
            if self.save_backgrounds:
                self.backgrounds[str(frame_num)] = self.background_filter.curr_background
        else:
            self.background_filter_remove_idx = []

        if self.save_filtered_objects:
            self.filtered_objects[str(frame_num)] = self.background_filter_remove_idx

        if self.background_filter is not None:
            filtered_detections = {}
            background_detections = {}
            filtered_detections["boxes"] = [detections["boxes"][i] for i in range(len(detections["boxes"]))
                                            if i not in self.filtered_objects[str(frame_num)]]
            background_detections["boxes"] = [detections["boxes"][i] for i in range(len(detections["boxes"]))
                                            if i in self.filtered_objects[str(frame_num)]]
            filtered_detections["scores"] = [detections["scores"][i] for i in
                                             range(len(detections["scores"])) if
                                             i not in self.filtered_objects[str(frame_num)]]
            background_detections["scores"] = [detections["scores"][i] for i in
                                                range(len(detections["scores"])) if
                                                i in self.filtered_objects[str(frame_num)]]
            filtered_detections["classes_num"] = [detections["classes_num"][i] for i in
                                                  range(len(detections["classes_num"])) if
                                                  i not in self.filtered_objects[str(frame_num)]]
            background_detections["classes_num"] = [detections["classes_num"][i] for i in
                                                    range(len(detections["classes_num"])) if
                                                    i in self.filtered_objects[str(frame_num)]]
            filtered_detections["classes"] = [detections["classes"][i] for i in
                                              range(len(detections["classes"])) if
                                              i not in self.filtered_objects[str(frame_num)]]
            background_detections["classes"] = [detections["classes"][i] for i in
                                                range(len(detections["classes"])) if
                                                i in self.filtered_objects[str(frame_num)]]
            filtered_detections_first_stage = self.filtered_objects[str(frame_num)]
        else:
            filtered_detections = detections
            background_detections = detections
            filtered_detections_first_stage = [q for q in range(len(detections["boxes"]))]

        if len(filtered_detections["boxes"]) == 0:
            return filtered_detections, np.array([])

        # Check if srrs_filter exists
        if self.srrs_filter is not None and len(background_detections["boxes"]) > 0:
            srrs_filter_remove_idx = self.srrs_filter.filter(background_detections,
                                                                  self.detector.get_detections_raw()[0])
            new_filter_idx = []
            for i, idx in enumerate(srrs_filter_remove_idx):
                new_filter_idx.append(filtered_detections_first_stage[i])
            self.srrs_filter_remove_idx = new_filter_idx
            self.filtered_objects[str(frame_num)] = new_filter_idx
            self.filtered_objects[str(frame_num)] = list(set(self.filtered_objects[str(frame_num)]))
            self.filtered_objects[str(frame_num)].sort()
        else:
            self.srrs_filter_remove_idx = []

        # Check if pbde_filter exists
        if self.pbde_filter is not None and len(background_detections["boxes"]) > 0:
            pbde_filter_remove_idx = self.pbde_filter.filter(background_detections,
                                                             self.detector.get_detections_raw()[0])
            new_filter_idx = []
            for i, idx in enumerate(pbde_filter_remove_idx):
                new_filter_idx.append(filtered_detections_first_stage[i])
            self.pbde_filter_remove_idx = new_filter_idx
            self.filtered_objects[str(frame_num)] = new_filter_idx
            self.filtered_objects[str(frame_num)] = list(set(self.filtered_objects[str(frame_num)]))
            self.filtered_objects[str(frame_num)].sort()
        else:
            self.pbde_filter_remove_idx = []

        detections_after_filter = {}
        detections_after_filter["boxes"] = list(
            np.delete(np.asarray(detections["boxes"]), self.filtered_objects[str(frame_num)], axis=0))
        detections_after_filter["scores"] = list(
            np.delete(np.asarray(detections["scores"]), self.filtered_objects[str(frame_num)], axis=0))
        detections_after_filter["classes_num"] = list(
            np.delete(np.asarray(detections["classes_num"]), self.filtered_objects[str(frame_num)], axis=0))
        detections_after_filter["classes"] = list(
            np.delete(np.asarray(detections["classes"]), self.filtered_objects[str(frame_num)], axis=0))

        dets_after_filter = np.delete(dets, self.filtered_objects[str(frame_num)], axis=0)


        #
        # # Check if pbde_filter exists
        # if self.pbde_filter is not None:
        #     detections_after_pbde_filter = self.pbde_filter.filter(frames, detections_after_srrs_filter)
        # else:
        #     detections_after_pbde_filter = detections_after_srrs_filter

        return detections_after_filter, dets_after_filter

    def has_background_filter(self):
        return self.background_filter is not None

    #def get_background_idx(self):
    #    return self.background_filter_remove_idx

    def get_filtered_objects(self, frame_num):
        if self.save_filtered_objects:
            return self.filtered_objects[str(frame_num)]
        else:
            return None

    def add_filtered_object(self, frame_num, filtered_object):
        # append filtered objects to filtered_objects
        self.filtered_objects[str(frame_num)].append(filtered_object)
        self.filtered_objects[str(frame_num)] = list(set(self.filtered_objects[str(frame_num)]))
        self.filtered_objects[str(frame_num)].sort()

    def get_background(self, frame_num):
        if self.save_backgrounds:
            return self.backgrounds[str(frame_num)]
        else:
            return None

    def set_save_filtered_objects(self, save_filtered_objects):
        self.save_filtered_objects = save_filtered_objects

    def set_save_backgrounds(self, save_backgrounds):
        self.save_backgrounds = save_backgrounds

    def adapt_filtered_objects(self,frame_num, det_idx):
        self.filtered_objects[str(frame_num)] = [x - 1 if x > det_idx else x for x in self.filtered_objects[str(frame_num)]]

    def clear_all(self):
        self.filtered_objects = {}
        self.backgrounds = {}
        if self.background_filter is not None:
            self.background_filter.clear_all()


