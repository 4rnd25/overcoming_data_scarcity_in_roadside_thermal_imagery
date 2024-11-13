"""
Created on Jan 12 11:00

@author: ISAC - pettirsch
"""
import pdb
# Imports
import time
import torch
import cv2
import numpy as np

from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, xywh2xyxy
from yolov7.utils.torch_utils import load_classifier, select_device, time_synchronized, TracedModel
from yolov7.utils.datasets import letterbox


class Detector_Yolo:

    def __init__(self, weights, img_size=640, conf_thesh=0.25, iou_thresh=0.65, classes=7, cls_agnostic_nms=True,
                 trace=True, device="0", get_raw_detections = False):

        self.conf_thresh = conf_thesh
        self.iou_thresh = iou_thresh
        self.classes = classes
        self.cls_agnostic_nms = cls_agnostic_nms
        self.img_size = img_size

        self.last_detections = None

        # Initialize
        try:
            self.device = select_device(device)
        except:
            self.device = select_device("cpu")
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(img_size, s=self.stride)  # check img_size

        if trace:
            self.model = TracedModel(model, self.device, img_size)

        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = model.module.names if hasattr(model, 'module') else model.names

        # Run inference
        if self.device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(model.parameters())))  # run once
        self.old_img_w = self.old_img_h = imgsz
        self.old_img_b = 1

        self.detections_raw = {}
        self.dets_raw = None

        self.get_raw_detections = get_raw_detections


    def detect(self, img0):

        batch_detections = []
        batch_detections_raw = []

        # Preprocess image
        # Padded resize
        img = letterbox(img0, new_shape=(self.img_size, self.img_size), auto=False, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (
                    self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=False)[0]

        # Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred_raw = self.model(img, augment=False)[0]

        # Apply NMS
        pred_after_nms = non_max_suppression(pred_raw, self.conf_thresh, self.iou_thresh, classes=None,
                                             agnostic=self.cls_agnostic_nms,
                                             multi_label=False)

        # Apply NMS without NMS
        if self.get_raw_detections:
            pred_raw = self.postprocess(pred_raw)

        # Process detections
        dets = []
        for i, det in enumerate(pred_after_nms):  # detections per image
            if len(det):
                det = det.cpu()
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                detections = {}
                detections["boxes"] = det[:, :4].numpy().tolist()
                detections["scores"] = det[:, 4].numpy().tolist()
                detections["classes_num"] = det[:, 5].numpy().tolist()
                detections["classes"] = [self.names[int(x)] for x in detections["classes_num"]]
                batch_detections.append(detections)

                # Extract boxes and scores from detections
                box_and_scores = np.hstack((det[:, :4], det[:, 4][:, np.newaxis]))
                # Check if boxes 0 equal boxes 2 and boxes 1 equal boxes 3
                mask = np.logical_and(
                    (box_and_scores[:, 0] != box_and_scores[:, 2]),
                    (box_and_scores[:, 1] != box_and_scores[:, 3])
                )
                # Append only those boxes that meet the condition
                dets.append(box_and_scores[mask])

                # Apply mask to detections dictionary
                detections["boxes"] = [bbox for bbox, m in zip(detections["boxes"], mask) if m]
                detections["scores"] = [score for score, m in zip(detections["scores"], mask) if m]
                detections["classes_num"] = [cls_num for cls_num, m in zip(detections["classes_num"], mask) if m]
                detections["classes"] = [cls for cls, m in zip(detections["classes"], mask) if m]
            else:
                detections = {}
                detections["boxes"] = []
                detections["scores"] = []
                detections["classes_num"] = []
                detections["classes"] = []
                batch_detections.append(detections)

                #box_and_scores = np.hstack((det[:, :4], det[:, 4][:, np.newaxis]))
                #dets = box_and_scores.astype(np.float32)
        dets = np.array(dets)
        if dets.shape[0] > 0:
            dets = dets[0,:,:]

        # Process raw detections
        if self.get_raw_detections:
            dets_raw = []
            for i, det in enumerate(pred_raw):
                if len(det):
                    det = det.cpu()
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                    detections = {}
                    detections["boxes"] = det[:, :4].numpy().tolist()
                    detections["scores"] = det[:, 4].numpy().tolist()
                    detections["classes_num"] = det[:, 5].numpy().tolist()
                    detections["classes"] = [self.names[int(x)] for x in detections["classes_num"]]
                    batch_detections_raw.append(detections)

                    # Extract boxes and scores from detections
                    box_and_scores = np.hstack((det[:, :4], det[:, 4][:, np.newaxis]))
                    dets_raw = box_and_scores.astype(np.float32)
            self.dets_raw = np.array(dets_raw)
            self.detections_raw = batch_detections_raw
        else:
            self.dets_raw = None
            self.detections_raw = {}

        return batch_detections, dets

    def postprocess(self, prediction):

        nc = prediction.shape[2] -5    # number of classes

        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction): # image index, image inference
            if not x.shape[0]:
                continue

            # Compute conf
            if nc == 1:
                x[:, 5:] = x[:, 4:5]  # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                # so there is no need to multiplicate.
            else:
                x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            i, j = (x[:, 5:] > -1).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue

            # Keep all boxes without applying NMS
            output[xi] = x

        return output

    def get_detections_raw(self):
        return self.detections_raw, self.dets_raw

    def set_get_raw_detections(self, get_raw_detections):
        self.get_raw_detections = get_raw_detections