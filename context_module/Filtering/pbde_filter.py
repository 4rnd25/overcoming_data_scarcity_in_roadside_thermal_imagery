"""
Created on Feb 15 2024 09:23

@author: ISAC - pettirsch
"""
import numpy as np

# create class PBDEFilter
class PBDEFilter:
    # create function __init__
    def __init__(self, pbde_thresh = 2):
        self.pbde_thresh = pbde_thresh

    # create function filter
    def filter(self, detections, detections_raw):
        det_boxes = np.asarray(detections["boxes"])
        raw_boxes = np.asarray(detections_raw[0]["boxes"])

        # Calculate the center of the detections
        det_centers = np.zeros((det_boxes.shape[0], 2))
        raw_centers = np.zeros((raw_boxes.shape[0], 2))
        det_centers[:, 0] = (det_boxes[:, 0] + det_boxes[:, 2]) / 2
        det_centers[:, 1] = (det_boxes[:, 1] + det_boxes[:, 3]) / 2
        raw_centers[:, 0] = (raw_boxes[:, 0] + raw_boxes[:, 2]) / 2
        raw_centers[:, 1] = (raw_boxes[:, 1] + raw_boxes[:, 3]) / 2

        # Calculate the center distance matrix
        dist_matrix = self._calc_dist_matrix(det_centers, raw_centers)

        # For each det_box calc r = min(x2-x1, y2-y1)
        r = np.minimum(det_boxes[:, 2] - det_boxes[:, 0], det_boxes[:, 3] - det_boxes[:, 1])
        r_thresh = r/2

        # For each det_box calc the number of raw_boxes that are within r_thresh
        n_r = np.zeros(det_boxes.shape[0])
        for i in range(det_boxes.shape[0]):
            n_r[i] = np.sum(dist_matrix[i] < r_thresh[i])

        return np.where(n_r < self.pbde_thresh)[0]

    def _calc_dist_matrix(self, curr_points, past_points):
        curr_points = curr_points[:, np.newaxis, :]
        past_points = past_points[np.newaxis, :, :]

        diff_matrix = curr_points - past_points
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)

        return dist_matrix





