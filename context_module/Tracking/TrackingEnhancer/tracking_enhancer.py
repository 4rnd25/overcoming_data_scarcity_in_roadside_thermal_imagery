"""
Created on Feb 07 2024 14:15

@author: ISAC - pettirsch
"""
import pdb

from context_module.Object_class.annotation_object import AnnotationObject


class TrackingEnhancer:
    def __init__(self, max_age=5, voting = None, min_track_length = 0, movement_required = False, movement_required_thresh=0,
                 save_history=False):

        self.max_age = max_age  # Maximum age of an object

        self.annotation_objects = {}  # Stores the objects that are annotated

        self.frame_object_dict = {}  # Stores the objects for each frame

        self.objects_finished = []  # Stores the objects that are finished

        self.change_history = {}  # Stores the changes of the objects
        self.save_history = save_history

        self.voting = voting # One of None, major, max_confidence, soft, weighted

        self.min_track_length = min_track_length
        self.movement_required = movement_required
        self.movement_required_thresh = movement_required_thresh

    def enhance_detections(self, detectionManger, frame_num, tracks, track_ids_intern, track_ids, video_finished=False,
                           final_call=False):

        detections = detectionManger.get_detections(frame_num)

        # Update annotation_objects
        if tracks is not None:
            self.update_annotation_objects(detections, frame_num, tracks, track_ids_intern, track_ids)

        finished_frames, finished_objects = self.filter_objects(frame_counter=frame_num,
                                                                remove_all=video_finished)

        filtered_objects = {}
        for obj in finished_objects:
            frames_changed = self.annotation_objects[obj].update_class(voting = self.voting)
            remove = self.annotation_objects[obj].check_remove(self.min_track_length, self.movement_required,
                                                               self.movement_required_thresh)
            if not remove:
                for frame_id in self.annotation_objects[obj].get_frame_nums():
                    idx = self.annotation_objects[obj].get_frame_nums().index(frame_id)
                    detectionManger.updateDetection(frame_id, int(float(obj)),
                                                    self.annotation_objects[obj].get_class_id()[idx],
                                                    self.annotation_objects[obj].get_box_2d()[idx],
                                                    self.annotation_objects[obj].get_confidence()[idx],
                                                    self.annotation_objects[obj].get_class_name()[idx])
                    if self.save_history and frame_id in frames_changed:
                        if frame_id not in self.change_history.keys():
                            self.change_history[frame_id] = []
                        self.change_history[frame_id].append(obj)
                    if frame_id in self.frame_object_dict.keys():
                        self.frame_object_dict[frame_id].remove(obj)
            else:
                frame_ids, filtered_detection = detectionManger.filterDetection(int(float(obj)))
                for q, frame_id in enumerate(frame_ids):
                    if frame_id not in filtered_objects.keys():
                        filtered_objects[frame_id] = []
                    filtered_objects[frame_id].append(filtered_detection[q])
                    if frame_id in self.frame_object_dict.keys():
                        self.frame_object_dict[frame_id].remove(obj)
            self.annotation_objects.pop(obj)

        # Check if detections are empty
        if len(detections["boxes"]) == 0 and not final_call:
            if frame_num not in finished_frames:
                finished_frames.append(frame_num)

        return finished_frames, filtered_objects


    def update_annotation_objects(self, detections, frame_num, tracks,track_ids_intern, track_ids):
        for det_idx, track_idx_glob in enumerate(detections["track_idx"]):
            idx = track_ids.index(track_idx_glob)
            track_idx_glob = str(int(float(track_idx_glob)))
            if str(track_idx_glob) not in self.annotation_objects.keys():
                print("Create new object: ", track_idx_glob)
                self.annotation_objects[track_idx_glob] = AnnotationObject()
            # Add detections to annotation_objects
            self.annotation_objects[track_idx_glob].add_annotation(frame=frame_num,
                                                                   class_id=detections["classes_num"][det_idx],
                                                                   bbox=tracks[track_ids_intern[det_idx], :4],
                                                                   confidence=detections["scores"][det_idx])

    def filter_objects(self, frame_counter, remove_all=False):

        self.update_dicts(self.annotation_objects, frame_counter)

        frames_to_create_labels = []
        keys_to_remove = []

        # remove objects which max frame_nums -self.max_age is smaller than frame_counter
        for obj_key in list(self.annotation_objects.keys()):
            if not remove_all:
                if max(self.annotation_objects[obj_key].get_frame_nums()) < (frame_counter - self.max_age):
                    keys_to_remove.append(obj_key)
                    self.objects_finished.append(obj_key)
            else:
                keys_to_remove.append(obj_key)
                self.objects_finished.append(obj_key)

        # Iterate through frame_object_dict check if all objects are finished
        frames_to_pop = []
        for frame_key in list(self.frame_object_dict.keys()):
            # Check is every object in frame_object_dict[frame_key] is in objects_finished
            if all(obj in self.objects_finished for obj in self.frame_object_dict[frame_key]):
                frames_to_create_labels.append(frame_key)
                frames_to_pop.append(frame_key)

        # Remove frames from frame_object_dict
        for frame in frames_to_pop:
            self.frame_object_dict.pop(frame)

        return frames_to_create_labels, keys_to_remove

    def update_dicts(self, objects_dict, frame_counter):
        for obj_key in objects_dict.keys():
            if objects_dict[obj_key].get_frame_nums()[-1] == frame_counter:
                # Check if str(frame_counter) is already in frame_object_dict keys
                if str(frame_counter) not in self.frame_object_dict:
                    self.frame_object_dict[str(frame_counter)] = []
                # Add object to frame_object_dict
                self.frame_object_dict[str(frame_counter)].append(obj_key)


    def get_changed_objects(self, frame_num):
        if frame_num in self.change_history.keys():
            return self.change_history[frame_num]
        else:
            return []

    def set_save_history(self, save_history):
        self.save_history = save_history

    def clear_all(self):
        keys_to_remove = []
        for ann_obj in self.annotation_objects.keys():
            keys_to_remove.append(ann_obj)
        for ann_obj in keys_to_remove:
            self.annotation_objects.pop(ann_obj)
        self.annotation_objects = {}
        self.frame_object_dict = {}
        self.objects_finished = []
        self.change_history = {}