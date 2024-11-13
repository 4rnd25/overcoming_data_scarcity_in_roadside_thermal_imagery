"""
Created on Jan 12 09:35

@author: ISAC - pettirsch
"""

# Import modules
import os.path
import argparse
import time
import json
import os

# Import own modules
from Utils.Loader.input_loading import input_loading
from context_module.Detection.detection_manager import DetectionManager
from context_module.Detection.detector_yolo import Detector_Yolo
from context_module.Filtering.Detection_Filter.detection_filter import DetectionFilter
from context_module.Tracking.Tracker.sort_tracker import SortTracker
from context_module.Tracking.TrackingEnhancer.tracking_enhancer import TrackingEnhancer
from context_module.Label_Creation.xml_label_creator import XML_Label_Creator
from context_module.Plotting.plotter import Plotter

# Arg parsesr parse parameter
parser = argparse.ArgumentParser()
parser.add_argument("--configPath", help="path to config file",
                    default="")
args = parser.parse_args()


def process_input(inputHandler, detector, detectionFilter=None,
                  tracker=None, trackingEnhancer=None, label_creator=None, ploter=None,
                  save_filter_and_tracking_stats = False, save_stats_path = None):

    if save_filter_and_tracking_stats:
        filter_tracking_stats = {"motorcycle": {"filtered": 0,
                                                "detected": 0},
                                 "car": {"filtered": 0,
                                         "detected": 0},
                                 "truck": {"filtered": 0,
                                           "detected": 0},
                                 "bus": {"filtered": 0,
                                         "detected": 0},
                                 "person": {"filtered": 0,
                                            "detected": 0},
                                 "bicycle": {"filtered": 0,
                                             "detected": 0},
                                 "e-scooter": {"filtered": 0,
                                               "detected": 0},
                                 "all": {"filtered": 0,
                                         "detected": 0}}
    else:
        filter_tracking_stats = None

    # Intiate detectionManager
    detectionManager = DetectionManager()

    # Load first frames
    if detectionFilter is not None and detectionFilter.has_background_filter():
        n_frames = detectionFilter.background_filter.window_size
    else:
        n_frames = 200000000
    if not inputHandler.is_video_folder_handler():
        inputHandler.load_first_frames(n_frames)
    else:
        inputHandler.set_first_frames(n_frames)

    # While not finished
    if detectionFilter is not None:
        if detectionFilter.has_background_filter():
            save_window_size = detectionFilter.background_filter.window_size
    while True:

        start = time.time()

        remove = False

        if detectionFilter is not None:
            if detectionFilter.has_background_filter():
                detectionFilter.background_filter.window_size = save_window_size
            else:
                remove = True

        # 1. Read frame
        frame, img_path = inputHandler.get_next_frame()

        if detectionFilter is not None:
            if detectionFilter.has_background_filter():
                if inputHandler.get_max_frames() < save_window_size:
                    detectionFilter.background_filter.window_size = inputHandler.get_max_frames()
                if inputHandler.get_max_frames() == save_window_size:
                    remove = True
            else:
                remove = True

        print_frame_num = inputHandler.get_frame_num() - 1
        print_max_frames = inputHandler.get_max_frames()

        if frame is None:
            if inputHandler.is_finished():
                break
            else:
                if inputHandler.is_video_folder_handler():
                    if inputHandler.getVideoIdx() == 0:
                        continue
                # Last enhancement
                if trackingEnhancer is not None:
                    finished_frames, filtered_objects_tracking = trackingEnhancer.enhance_detections(detectionManager,
                                                                                                     frame_num=inputHandler.get_frame_num() - 1,
                                                                                                     tracks=curr_tracks,
                                                                                                     track_ids_intern=track_idx_intern_list,
                                                                                                     track_ids=track_idx_list,
                                                                                                     video_finished=True,
                                                                                                     final_call = True)
                    for frame_id in filtered_objects_tracking.keys():
                        if detectionFilter is not None:
                            for filtered_object in filtered_objects_tracking[frame_id]:
                                detectionFilter.add_filtered_object(frame_id, filtered_object)
                else:
                    finished_frames = []

                # Create final labels
                if label_creator is not None or ploter is not None:
                    for frame_id in finished_frames:
                        if label_creator is not None:
                            if ploter is not None:
                                frame = inputHandler.get_frame_by_id(int(frame_id), remove=False)
                            else:
                                frame = inputHandler.get_frame_by_id(int(frame_id), remove=remove)
                            if frame is not None:
                                label_creator.create_labels(detectionManager.get_imagePath(int(frame_id)),
                                                            detectionManager.get_detections(int(frame_id)),
                                                            frame_num=int(frame_id))
                        if ploter is not None:
                            frame = inputHandler.get_frame_by_id(int(frame_id), remove=remove)
                            if frame is not None:
                                ploter.plot_flex(frame, detectionManager.get_detections_raw(int(frame_id)),
                                                 detectionFilter=detectionFilter,
                                                 trackingEnhancer=trackingEnhancer,
                                                 image_name=detectionManager.get_imagePath(int(frame_id)),
                                                 frame_num=int(frame_id),
                                                 waitkey=33)

                    filter_tracking_stats=  detectionManager.fill_stats(filter_tracking_stats)

                    detectionManager.clear_all()
                    if detectionFilter is not None:
                        detectionFilter.clear_all()
                    if tracker is not None:
                        tracker.clear_all()
                    if trackingEnhancer is not None:
                        trackingEnhancer.clear_all()
                    finished_frames = []
                continue

        # 2. Detect on Frame
        detections, dets = detector.detect(frame)
        print_time_detection = round((time.time() - start)*1000,2)
        print_num_obj_det = len(detections[0]["boxes"])

        # 3. Filter Detections
        if detectionFilter is not None:
            detections_filtered, dets_filtered  = detectionFilter.filter(detections, dets,
                                                                        frame_num=inputHandler.get_frame_num() - 1,
                                                                        frames=inputHandler.get_frames())
        else:
            detections_filtered = detections[0]
            dets_filtered = dets
        print_num_after_filter = len(detections_filtered["boxes"])
        print_time_filter = round((time.time() - start)*1000,2)

        # 4. Tracking
        if tracker is not None and len(detections_filtered["boxes"]) > 0:
            track_idx_list, dets_without_track, curr_tracks, track_idx_intern_list = tracker.track(detections_filtered,
                                                                                                   dets_filtered,
                                                                                                   frame_num=inputHandler.get_frame_num() - 1)
            detections_filtered["track_idx"] = track_idx_list
            # Sort dets_without_track beginning with the highest score
            dets_without_track = sorted(dets_without_track, key=lambda x: detections_filtered["scores"][x],
                                        reverse=False)
            for det_idx in dets_without_track:
                detections_filtered["boxes"].pop(det_idx)
                detections_filtered["scores"].pop(det_idx)
                detections_filtered["classes_num"].pop(det_idx)
                detections_filtered["classes"].pop(det_idx)
                # detelete raw detection as well
                if detectionFilter is not None:
                    filtered_objects = detectionFilter.get_filtered_objects(inputHandler.get_frame_num() - 1)
                    # Count how many filtered_detection have index smaller than det_idx
                    count = 0
                    for filtered_idx in filtered_objects:
                        if filtered_idx <= det_idx:
                            count += 1
                    detections[0]["boxes"].pop(det_idx + count)
                    detections[0]["scores"].pop(det_idx + count)
                    detections[0]["classes_num"].pop(det_idx + count)
                    detections[0]["classes"].pop(det_idx + count)

                    detectionFilter.adapt_filtered_objects(inputHandler.get_frame_num() - 1, det_idx + count)
                    # detectionFilter.add_filtered_object(inputHandler.get_frame_num()-1, det_idx + count)
        else:
            detections_filtered["track_idx"] = []
            curr_tracks = None
            track_idx_intern_list = None
            track_idx_list = None
            if tracker is not None:
                tracker.check_remove(inputHandler.get_frame_num() - 1)
        print_time_tracking = round((time.time() - start)*1000,2)

        # 5. Update detectionManager
        detections[0]["track_idx"] = detections_filtered["track_idx"]
        if detectionFilter is not None:
            detectionManager.update(detections[0], detections_filtered, inputHandler.get_frame_num() - 1,
                                    imagePath=img_path,
                                    filterIdx=detectionFilter.get_filtered_objects(inputHandler.get_frame_num() - 1))
        else:
            detectionManager.update(detections[0], detections_filtered, inputHandler.get_frame_num() - 1,
                                    imagePath=img_path)

        # 6. Tracking Enhancer
        if trackingEnhancer is not None:
            finished_frames, filtered_objects_tracking = trackingEnhancer.enhance_detections(detectionManager,
                                                                                             frame_num=inputHandler.get_frame_num() - 1,
                                                                                             tracks=curr_tracks,
                                                                                             track_ids_intern=track_idx_intern_list,
                                                                                             track_ids=track_idx_list,
                                                                                             video_finished=inputHandler.is_finished())
            for frame_id in filtered_objects_tracking.keys():
                if detectionFilter is not None:
                    for filtered_object in filtered_objects_tracking[frame_id]:
                        detectionFilter.add_filtered_object(frame_id, filtered_object)
        else:
            finished_frames = [inputHandler.get_frame_num() - 1]
        print_time_tracking_enhancer = round((time.time() - start)*1000,2)

        # 7., 8. Create labels and Plot
        if label_creator is not None or ploter is not None:
            for frame_id in finished_frames:
                if label_creator is not None:
                    label_creator.create_labels(detectionManager.get_imagePath(int(frame_id)),
                                                detectionManager.get_detections(int(frame_id)),
                                                frame_num=int(frame_id))
                if ploter is not None:
                    frame = inputHandler.get_frame_by_id(int(frame_id), remove=remove)
                    ploter.plot_flex(frame, detectionManager.get_detections_raw(int(frame_id)),
                                     detectionFilter=detectionFilter,
                                     trackingEnhancer=trackingEnhancer,
                                     image_name=detectionManager.get_imagePath(int(frame_id)),
                                     frame_num=int(frame_id),
                                     waitkey=33)


        # 9. print
        print("\nProcessing frame {} from {}\n"
              "Detection: {}ms, {} objects\n"
              "Filter: {}ms, {} objects\n"
              "Tracking: {}ms, Tracking Enhancer: {}ms".format(
            print_frame_num, print_max_frames, print_time_detection, print_num_obj_det, print_time_filter,
            print_num_after_filter, print_time_tracking, print_time_tracking_enhancer),end='', flush=True)
        print("\r", end='', flush=True)

    # Last enhancement
    if not detectionManager.is_empty():
        if trackingEnhancer is not None:
            finished_frames, filtered_objects_tracking = trackingEnhancer.enhance_detections(detectionManager,
                                                                                             frame_num=inputHandler.get_frame_num() - 1,
                                                                                             tracks=curr_tracks,
                                                                                             track_ids_intern=track_idx_intern_list,
                                                                                             track_ids=track_idx_list,
                                                                                             video_finished=True)
            for frame_id in filtered_objects_tracking.keys():
                if detectionFilter is not None:
                    for filtered_object in filtered_objects_tracking[frame_id]:
                        detectionFilter.add_filtered_object(frame_id, filtered_object)

        # Create final labels
        if label_creator is not None or ploter is not None:
            for frame_id in finished_frames:
                if label_creator is not None:
                    label_creator.create_labels(detectionManager.get_imagePath(int(frame_id)),
                                                detectionManager.get_detections(int(frame_id)),
                                                frame_num=int(frame_id))
                if ploter is not None:
                    frame = inputHandler.get_frame_by_id(int(frame_id), remove=remove)
                    ploter.plot_flex(frame, detectionManager.get_detections_raw(int(frame_id)),
                                     detectionFilter=detectionFilter,
                                     trackingEnhancer=trackingEnhancer,
                                     image_name=detectionManager.get_imagePath(int(frame_id)),
                                     frame_num=int(frame_id),
                                     waitkey=33)

    if ploter is not None:
        ploter.destroy_all_windows()

    if save_filter_and_tracking_stats:
        if save_stats_path is not None:
            filename = "filter_tracking_stats.json"
            save_stats_path = os.path.join(save_stats_path, filename)
            with open(save_stats_path, "w") as write_file:
                json.dump(filter_tracking_stats, write_file)


def intiate_pseudo_label_creation(inputPath):
    # Read in json config file as dictionary
    with open(args.configPath, "r") as read_file:
        config = json.load(read_file)

    # Initiate input handler
    inputHandler = input_loading(config["input_config"]["input_path"])

    # Load detector
    detector = Detector_Yolo(weights=config["detection_config"]["weights_path"],
                             img_size=config["detection_config"]["img_size"],
                             conf_thesh=config["detection_config"]["conf_thesh"],
                             iou_thresh=config["detection_config"]["iou_thresh"],
                             classes=config["detection_config"]["classes"],
                             cls_agnostic_nms=config["detection_config"]["cls_agnostic_nms"],
                             trace=config["detection_config"]["trace"], device=config["detection_config"]["device"])

    # Check if filter_config is in config
    if "detection_filter_config" in config:
        detectionFilter = DetectionFilter(background_filter=config["detection_filter_config"]["background_filter"],
                                          background_filter_config=config["detection_filter_config"][
                                              "background_filter_config"],
                                          srrs_filter=config["detection_filter_config"]["srrs_filter"],
                                          srrs_filter_config=config["detection_filter_config"]["srrs_filter_config"],
                                          pbde_filter=config["detection_filter_config"]["pbde_filter"],
                                          pbde_filter_config=config["detection_filter_config"]["pbde_filter_config"],
                                          detector=detector)
    else:
        detectionFilter = None

    # Check if tracking_config is in config
    if "tracking_config" in config:
        if config["tracking_config"]["tracker"] == "SORT":
            tracker = SortTracker(max_age=config["tracking_config"]["max_age"],
                                  iou_threshold=config["tracking_config"]["iou_threshold"],
                                  min_hits=config["tracking_config"]["min_hits"])
        else:
            raise NotImplementedError("Tracker not implemented")
    else:
        tracker = None

    # Check if tracking_enhancer_config is in config
    if "tracking_enhancer_config" in config:
        assert tracker is not None, "Tracker must be initialized to use tracking enhancer"
        trackingEnhancer = TrackingEnhancer(config["tracking_enhancer_config"]["max_age"],
                                            config["tracking_enhancer_config"]["voting"],
                                            config["tracking_enhancer_config"]["min_track_length"],
                                            config["tracking_enhancer_config"]["movement_required"],
                                            config["tracking_enhancer_config"]["movement_required_thresh"])
    else:
        trackingEnhancer = None

    # Check if label_creator_config is in config
    if "label_creator_config" in config:
        if config["label_creator_config"]["output_format"] == "XML":
            if "output_only_if_detections_exist" in config["label_creator_config"]:
                output_only_if_detections_exist = config["label_creator_config"]["output_only_if_detections_exist"]
            else:
                output_only_if_detections_exist = False
            label_creator = XML_Label_Creator(config["label_creator_config"]["output_path"],
                                              config["label_creator_config"]["add_score"],
                                              config["label_creator_config"]["label_every_nth_frame"],
                                              output_only_if_detections_exist=output_only_if_detections_exist)
        else:
            raise NotImplementedError("Output format not implemented")
    else:
        label_creator = None

    # Check if plot_config is in config
    if "plot_config" in config:
        if "raw_images_path" in config["plot_config"]:
            ploter = Plotter(output_path=config["plot_config"]["output_path"],
                             show_output=config["plot_config"]["show_output"],
                             plot_every_nth_frame=config["label_creator_config"]["label_every_nth_frame"],
                             raw_images_path=config["plot_config"]["raw_images_path"])
        else:
            ploter = Plotter(output_path=config["plot_config"]["output_path"],
                             show_output=config["plot_config"]["show_output"],
                             plot_every_nth_frame=config["label_creator_config"]["label_every_nth_frame"],
                             raw_images_path=None)
        if trackingEnhancer is not None:
            trackingEnhancer.set_save_history(True)
        if detectionFilter is not None:
            detectionFilter.set_save_filtered_objects(True)
            if detectionFilter.has_background_filter():
                detectionFilter.set_save_backgrounds(True)
    else:
        ploter = None

    # Process input
    process_input(inputHandler=inputHandler, detector=detector, detectionFilter=detectionFilter,
                  tracker=tracker, trackingEnhancer=trackingEnhancer, label_creator=label_creator, ploter=ploter,
                  save_filter_and_tracking_stats=config["stats_config"]["save_stats"],
                  save_stats_path=config["stats_config"]["save_stats_path"])


if __name__ == "__main__":
    intiate_pseudo_label_creation(args.configPath)
