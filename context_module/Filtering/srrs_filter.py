"""
Created on Feb 15 2024 08:26

@author: ISAC - pettirsch
"""
import numpy as np

from Pseudo_Label_Creation.Utils.matching_utils import calc_box_iou_matrix


# create class SRRSFilter
class SRRSFilter:
    # create function __init__
    def __init__(self, iou_thresh = 0.5, srrs_thresh = 0.8):
        # Create pbde_filter_config
        self.iou_thresh = iou_thresh
        self.srrs_thresh = srrs_thresh

    # create function filter
    def filter(self, detections, detections_raw):
        # Create list of filtered detections
        det_boxes = np.asarray(detections["boxes"])
        raw_boxes = np.asarray(detections_raw[0]["boxes"])

        iou_matrix = calc_box_iou_matrix(det_boxes, raw_boxes)

        srrs_scores = []
        for i in range(iou_matrix.shape[0]):
            # get all indices where iou_matrix[i] > iou_thresh
            iou_indices = np.where(iou_matrix[i] > self.iou_thresh)[0]
            n_s = iou_indices.shape[0]
            srrs_score = 0
            for j, idx in enumerate(list(iou_indices)):
                if detections_raw[0]["classes_num"][idx] == detections["classes_num"][i]:
                    srrs_score += (iou_matrix[i, idx]*detections_raw[0]["scores"][idx])
            if n_s != 0:
                srrs_score = srrs_score/n_s
            srrs_scores.append(srrs_score)

        dets_to_remove = []
        for i, srrs_score in enumerate(srrs_scores):
            if srrs_score < self.srrs_thresh:
                dets_to_remove.append(i)

        return dets_to_remove