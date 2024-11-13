"""
Created on Feb 15 2024 11:23

@author: ISAC - pettirsch
"""
import numpy as np
import pdb

class DetectionManager:
    def __init__(self):
        self.detections_filtered = {}
        self.filterIdx = {}
        self.detections = {}
        self.imagePaths = {}

    def update(self, detections, detections_filtered, frame_num, imagePath=None, filterIdx=None):
        self.detections[str(frame_num)] = detections
        self.detections_filtered[str(frame_num)] = detections_filtered
        self.imagePaths[str(frame_num)] = imagePath
        if filterIdx is not None:
            if str(frame_num) not in self.filterIdx.keys():
                self.filterIdx[str(frame_num)] = []
            self.filterIdx[str(frame_num)].extend(filterIdx)

    def updateDetection(self, frame_id, obj, class_id, bbox, confidence, class_name):
        idx = self.detections_filtered[str(frame_id)]["track_idx"].index(obj)
        if idx != -1 and idx is not None:
            if self.detections_filtered[str(frame_id)]["classes_num"][idx] != class_id:
                if str(frame_id) not in self.filterIdx.keys():
                    self.filterIdx[str(frame_id)] = []
                self.filterIdx[str(frame_id)].append(idx)
            self.detections_filtered[str(frame_id)]["classes_num"][idx] = class_id
            self.detections_filtered[str(frame_id)]["boxes"][idx] = bbox
            self.detections_filtered[str(frame_id)]["scores"][idx] = confidence
            self.detections_filtered[str(frame_id)]["classes"][idx] = class_name

    def filterDetection(self, obj):
        frame_ids = []
        filtered_detection = []

        for frame_id in self.detections_filtered.keys():
            if obj in self.detections_filtered[frame_id]["track_idx"]:
                idx_filtered = self.detections_filtered[str(frame_id)]["track_idx"].index(obj)
                box = np.asarray(self.detections_filtered[str(frame_id)]["boxes"][idx_filtered]).tolist()
                if box in self.detections[str(frame_id)]["boxes"]:
                    idx = self.detections[str(frame_id)]["boxes"].index(box)
                else:
                    #box = [float(int(round(x,0))) for x in box]
                    if box in self.detections[str(frame_id)]["boxes"]:
                        idx = self.detections[str(frame_id)]["boxes"].index(box)
                    else:
                        box = [float(int(round(x, 0))) for x in box]
                        if box in self.detections[str(frame_id)]["boxes"]:
                            idx = self.detections[str(frame_id)]["boxes"].index(box)
                        else:
                            pdb.set_trace()
                frame_ids.append(frame_id)
                if str(frame_id) not in self.filterIdx.keys():
                    self.filterIdx[str(frame_id)] = []
                self.filterIdx[str(frame_id)].append(idx)
                self.detections_filtered[frame_id]["track_idx"].pop(idx_filtered)
                self.detections_filtered[frame_id]["classes_num"].pop(idx_filtered)
                self.detections_filtered[frame_id]["boxes"].pop(idx_filtered)
                self.detections_filtered[frame_id]["scores"].pop(idx_filtered)
                self.detections_filtered[frame_id]["classes"].pop(idx_filtered)
                filtered_detection.append(idx)
        return frame_ids, filtered_detection

    def get_imagePath(self, frame_id):
        return self.imagePaths[str(frame_id)]

    def get_detections(self, frame_id):
        return self.detections_filtered[str(frame_id)]

    def get_detections_raw(self, frame_id):
        return self.detections[str(frame_id)]

    def clear_all(self):
        self.detections_filtered = {}
        self.detections = {}
        self.imagePaths = {}
        self.filterIdx = {}

    def is_empty(self):
        return len(self.detections) == 0

    def fill_stats(self, stats):
        for frame_id in self.detections.keys():
            if str(frame_id) in self.filterIdx.keys():
                filtered_objects = self.filterIdx[str(frame_id)]
            else:
                filtered_objects = []

            for id, class_name in enumerate(self.detections[frame_id]["classes"]):
                if id in filtered_objects:
                    stats[class_name]["filtered"] += 1
                    stats["all"]["filtered"] += 1
                else:
                    stats[class_name]["detected"] += 1
                    stats["all"]["detected"] += 1

        return stats