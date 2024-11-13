"""
Created on Feb 16 2024 13:49

@author: ISAC - pettirsch
"""
import pdb
import os
import cv2
import numpy as np


class Plotter():
    # Constructor
    def __init__(self, output_path=None, show_output=False, plot_every_nth_frame = 15, video_name="Output_Video",
                 raw_images_path=None):
        self.color_id_dict = {}
        self.output_path = output_path
        self.show_output = show_output
        self.video_name = video_name
        self.plot_every_nth_frame = plot_every_nth_frame
        self.raw_image_path = raw_images_path

    def plot_flex(self, frame, detections, detectionFilter, trackingEnhancer, image_name, frame_num, waitkey=1):
        if frame_num % self.plot_every_nth_frame != 0 and frame_num != 0:
            return

        background = None
        changed_objects = []
        if detectionFilter is not None:
            filtered_indices = detectionFilter.get_filtered_objects(frame_num)
            if detectionFilter.has_background_filter():
                background = detectionFilter.get_background(frame_num)
        else:
            filtered_indices = []
        if trackingEnhancer is not None:
            changed_objects = trackingEnhancer.get_changed_objects(frame_num)

        self.plot(frame, detections, filtered_indices, changed_objects, background, image_name, waitkey=waitkey, frame_num=frame_num)

    def plot(self, frame, detections, filtered_indices=[], changed_objects=[], background=None,
             image_name="frame", waitkey=1, frame_num=0):

        if self.raw_image_path is not None:
            filename = image_name.split("/")[-1]
            filename_split = filename.split("_")
            filename_new = ""
            for i in range(len(filename_split) - 1):
                filename_new += filename_split[i] + "_"
            filename_img = filename_new[:-1] + "_" + str(frame_num) + ".png"
            cv2.imwrite(os.path.join(self.raw_image_path, filename_img), frame)

        # draw detections and labels
        valid_dets_counter = 0
        for i, box in enumerate(detections["boxes"]):
            if i in filtered_indices:
                color = (0, 0, 255)
                label = str(detections["classes"][i]) + " " + str(round(detections["scores"][i], 2)) + " " + "filtered"
            else:
                if len(detections["track_idx"]) > 0:
                    try:
                        track_idx = detections["track_idx"][valid_dets_counter]
                    except:
                        # pdb.set_trace()
                        print("continue")
                    valid_dets_counter += 1
                    if track_idx in changed_objects:
                        color = (0, 255, 0)
                    else:
                        if track_idx in self.color_id_dict.keys():
                            color = self.color_id_dict[track_idx]
                        else:
                            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                            self.color_id_dict[track_idx] = color
                else:
                    color = (0, 255, 0)
                    track_idx = ""
                label = str(detections["classes"][i]) + " " + str(round(detections["scores"][i], 2)) + " " + str(
                    track_idx)
            frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            frame = cv2.putText(frame, label, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if background is not None:
            frame = np.hstack((frame, background))

        if self.output_path is not None:
            # Get only filename from image_name
            filename = image_name.split("/")[-1]
            filename_split = filename.split("_")
            filename_new = ""
            for i in range(len(filename_split) - 1):
                filename_new += filename_split[i] + "_"
            filename_img = filename_new[:-1] + "_" + str(frame_num) + ".png"

            cv2.imwrite(os.path.join(self.output_path,filename_img), frame)

        if self.show_output:
            cv2.imshow(self.video_name, frame)

            cv2.waitKey(waitkey)

    def destroy_all_windows(self):
        if self.show_output:
            cv2.destroyAllWindows()
