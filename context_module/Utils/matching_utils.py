"""
Created on Feb 15 2024 08:21

@author: ISAC - pettirsch
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersection_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    intersection_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection_area = intersection_x * intersection_y

    union_area = w1 * h1 + w2 * h2 - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def match_dets(dets_1, dets_2, iou_threshold=0.5):
    """
    Match each row of dets to a track ID in tracks based on IoU.
    """
    # Extract box coordinates from detections
    boxes_1 = dets_1[:, :4]
    boxes_2 = dets_2[:, :4]

    # Calculate intersection areas
    intersection_areas = calculate_intersection_area(boxes_1, boxes_2)

    # Calculate box areas
    box_areas_1 = calculate_box_area(boxes_1)
    box_areas_2 = calculate_box_area(boxes_2)

    # Calculate union areas
    union_areas = box_areas_1[:, np.newaxis] + box_areas_2 - intersection_areas

    # Calculate IoU matrix
    iou_matrix = intersection_areas / union_areas

    # Use linear sum assignment to match bounding boxes based on IoU
    matched_rows, matched_cols = linear_sum_assignment(-iou_matrix)  # Minimize negative IoU

    # Filter out matches with IoU less than 0.5
    valid_matches = iou_matrix[matched_rows, matched_cols] >= iou_threshold
    matched_rows = matched_rows[valid_matches]
    matched_cols = matched_cols[valid_matches]

    # STack matched rows and matched cols to have shape [(matched_row, matched_col), ...]
    matched_indices = np.stack((matched_rows, matched_cols), axis=1)
    matched_indices = matched_indices.tolist()

    return matched_indices

def calc_box_iou_matrix(boxes1, boxes2):
    """
    Calculate the IoU matrix between two sets of boxes.
    """
    intersection_areas = calculate_intersection_area(boxes1, boxes2)
    box_areas_1 = calculate_box_area(boxes1)
    box_areas_2 = calculate_box_area(boxes2)

    union_areas = box_areas_1[:, np.newaxis] + box_areas_2 - intersection_areas

    iou_matrix = intersection_areas / union_areas

    return iou_matrix

def calculate_intersection_area(boxes1, boxes2):
    """
    Calculate the intersection area between two sets of boxes.
    """
    x1 = np.maximum(boxes1[:, 0][:, np.newaxis], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, 1][:, np.newaxis], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, 2][:, np.newaxis], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, 3][:, np.newaxis], boxes2[:, 3])

    intersection_widths = np.maximum(0, x2 - x1)
    intersection_heights = np.maximum(0, y2 - y1)

    return intersection_widths * intersection_heights

def calculate_box_area(boxes):
    """
    Calculate the area of each box.
    """
    widths = np.maximum(0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0, boxes[:, 3] - boxes[:, 1])

    return widths * heights
