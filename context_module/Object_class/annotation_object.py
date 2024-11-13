"""
Created on Feb 05 2024 10:11

@author: ISAC - pettirsch
"""
import math

class AnnotationObject:
    def __init__(self):
        self.frame_nums = []
        self.box_2d = []
        self.class_name = []
        self.class_id = []
        self.confidence = []
        self.id_class_dict = {"0": "motorcycle",
                              "1": "car",
                              "2": "truck",
                              "3": "bus",
                              "4": "person",
                              "5": "bicycle",
                              "6": "e-scooter"}

    # Method to add an annotation
    def add_annotation(self, frame, class_id, bbox, confidence):
        self.frame_nums.append(frame)
        self.box_2d.append(bbox)
        self.class_id.append(class_id)
        self.class_name.append(self.id_class_dict[str(int(class_id))])
        self.confidence.append(confidence)

    # Update class
    def update_class(self, voting="major"):

        frames_changed = []

        if voting == "major":
            # Get class with highest number of occurences
            class_id = max(set(self.class_id), key=self.class_id.count)
            # Update class_id and class_name
            for i in range(len(self.class_id)):
                if self.class_id[i] != class_id:
                    frames_changed.append(self.frame_nums[i])
                self.class_id[i] = class_id
                self.class_name[i] = self.id_class_dict[str(int(class_id))]
        elif voting == "max_confidence":
            # Get class with highest confidence
            class_id = self.class_id[self.confidence.index(max(self.confidence))]
            # Update class_id and class_name
            for i in range(len(self.class_id)):
                if self.class_id[i] != class_id:
                    frames_changed.append(self.frame_nums[i])
                self.class_id[i] = class_id
                self.class_name[i] = self.id_class_dict[str(int(class_id))]
        elif voting == "soft":
            id_list = []
            scores_lists = []
            for i, id in enumerate(self.class_id):
                if id not in id_list:
                    id_list.append(id)
                    scores_lists.append([self.confidence[i]])
                else:
                    scores_lists[id_list.index(id)].append(self.confidence[i])

            # Calculate mean confidence for each class
            mean_confidences = [sum(scores) / len(scores) for scores in scores_lists]

            # Get class with highest mean confidence
            class_id = id_list[mean_confidences.index(max(mean_confidences))]

            # Update class_id and class_name
            for i in range(len(self.class_id)):
                if self.class_id[i] != class_id:
                    frames_changed.append(self.frame_nums[i])
                self.class_id[i] = class_id
                self.class_name[i] = self.id_class_dict[str(int(class_id))]
        elif voting == "weighted":
            # Calculate box areas
            box_areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in self.box_2d]

            id_list = []
            area_lists = []
            for i, id in enumerate(self.class_id):
                if id not in id_list:
                    id_list.append(id)
                    area_lists.append([box_areas[i]])
                else:
                    area_lists[id_list.index(id)].append(box_areas[i])

            # Calculate sum of areas for each class
            sum_areas = [sum(areas) for areas in area_lists]

            # Get class with highest sum of areas
            class_id = id_list[sum_areas.index(max(sum_areas))]

            # Update class_id and class_name
            for i in range(len(self.class_id)):
                if self.class_id[i] != class_id:
                    frames_changed.append(self.frame_nums[i])
                self.class_id[i] = class_id
                self.class_name[i] = self.id_class_dict[str(int(class_id))]
        else:
            frames_changed = []

        return frames_changed

    def check_remove(self, min_track_length=0, movement_required=False, movement_required_thresh=0.0):
        if len(self.frame_nums) < min_track_length:
            return True
        else:
            if movement_required:
                # Calculate the distance between the first and last box center
                center_1 = [(self.box_2d[0][0] + self.box_2d[0][2]) / 2, (self.box_2d[0][1] + self.box_2d[0][3]) / 2]
                center_2 = [(self.box_2d[-1][0] + self.box_2d[-1][2]) / 2, (self.box_2d[-1][1] + self.box_2d[-1][3]) / 2]
                min_movement_distance = min(self.box_2d[0][2] - self.box_2d[0][0], self.box_2d[0][3] - self.box_2d[0][1])/2
                center_distance = math.sqrt((center_2[0] - center_1[0]) ** 2 + (center_2[1] - center_1[1]) ** 2)
                if center_distance < min_movement_distance:
                    if min(self.confidence) < movement_required_thresh:
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False

    # Get methods
    def get_frame_nums(self):
        return self.frame_nums

    def get_box_2d(self):
        return self.box_2d

    def get_class_name(self):
        return self.class_name

    def get_class_id(self):
        return self.class_id

    def get_confidence(self):
        return self.confidence
