"""
Created on Feb 06 2024 13:05

@author: ISAC - pettirsch
"""

import argparse
import pdb
from pathlib import Path
import torch
import yaml
import numpy as np
import xml.etree.ElementTree as ET
import os
import json
from threading import Thread

from yolov7.utils.general import check_file, increment_path, check_dataset, xywh2xyxy, box_iou
from yolov7.utils.metrics import ap_per_class, ConfusionMatrix
from yolov7.utils.plots import plot_images, output_to_target

def test_pseudo_labels(data,
                       pseudo_label_path,
                       extra_conf_thres,
                       img_width=640,
                       img_height=480,
                       plots=True,
                       device="cpu",
                       ignore_files_without_detection=False,
                       verbose = True):

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    if isinstance(pseudo_label_path, list):
        print("pseudo_label_path is list")
        pseudo_label_path = pseudo_label_path[0]
    print("Psuedo label path: ", pseudo_label_path)

    # Configure
    if isinstance(data, str):
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Intialize parameters
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    # Initialize variables for TP, FP, FN, TN
    TP, FP, FN, TN = 0, 0, 0, 0

    # Replace "images" with "labels" in data in key 'opt.task'
    data_label_path = data[opt.task].replace('images', 'labels')

    # Create a list with all .txt files in data_label_path
    data_label_files = [f for f in Path(data_label_path).rglob("*.txt")]
    batch_i = 0
    image_path_list = []
    for file_counter, label_file in enumerate(data_label_files):
        data_list = []
        # Open the text file and read line by line
        with open(label_file, "r") as file:
            for line in file:
                # Split the line by whitespace
                values = [float(val) for val in line.split()]
                # Add 1 to the beginning of the list
                values.insert(0, 0)
                # Append the values to the data_list
                data_list.append(values)

        # Convert the data_list to numpy array #cls x,y,w,h
        data_array = np.array(data_list)

        # Convert the data_array to tensor
        targets = torch.tensor(data_array)
        targets.to(device)

        # Read in the json annotation file
        # Create tensor of shape n x 6 with conf, x1, y1, x2, y2, class for each detection from xml
        pseudo_label_file = pseudo_label_path + "/" + label_file.name.replace(".txt", ".xml")

        if ".mp4" in pseudo_label_file:
            pseudo_label_file = pseudo_label_file.replace(".mp4", "")
        image_path, labels, boxes, height, width, scores = parse_xml(pseudo_label_file)
        image_path_list.append(image_path)
        print("Len image_path_list: ", len(image_path_list))
        out = torch.zeros((len(labels), 6))
        for i, (label, box, score) in enumerate(zip(labels, boxes, scores)):
            out[i, 0] = box[0]
            out[i, 1] = box[1]
            out[i, 2] = box[2]
            out[i, 3] = box[3]
            out[i, 4] = score
            out[i, 5] = label
        # expand first dimension of out
        out = [out]

        ignore_files_without_detection = True
        if ignore_files_without_detection:
            if out[0].shape[0]== 0:
                continue


        # Multiply all objects in column 3 to 6 of targets elementwise with the image width and height
        targets[:, 2:] *= torch.tensor([width, height, width, height])

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictionsyolov7.
            predn = pred.clone()

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1) # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            #Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            #f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            #Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

        batch_i += 1

    # Compute statistics
    names = {0: "motorcycle", 1: "car", 2: "truck", 3: "bus", 4: "person", 5: "bicycle", 6: "e-scooter"}
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

    # Create a list with all unique true and predicted classes
    unique_true_classes = np.unique(stats[3])
    unique_pred_classes = np.unique(stats[2])

    # Create a list with all unique classes
    unique_classes = np.unique(np.concatenate((unique_true_classes, unique_pred_classes)))

    names_new = {}

    # sort unique classes beginning with smallest number
    unique_classes.tolist()
    unique_classes.sort()
    for i, cls in enumerate(unique_classes):
        names_new[i] = names[int(cls)]

    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=False, save_dir=save_dir,
                                              names=names_new)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Compute extra statistics at fixed score threshold
    conf = stats[1]  # confidence
    filter_inds = np.where(conf > extra_conf_thres)
    correct = stats[0]
    print(correct.shape)
    filtered_stats = [stats[0][:, 0][filter_inds], stats[1][filter_inds], stats[2][filter_inds], stats[3]]

    # Get the unique classes
    unique_classes = np.unique(filtered_stats[3])
    # Initialize dictionaries to store the TP, FP, FN, precision, recall, and F1-score for each class
    TP_dict = {names[int(cls)]: 0 for cls in unique_classes}
    FP_dict = {names[int(cls)]: 0 for cls in unique_classes}
    FN_dict = {names[int(cls)]: 0 for cls in unique_classes}
    precision_dict = {names[int(cls)]: 0 for cls in unique_classes}
    recall_dict = {names[int(cls)]: 0 for cls in unique_classes}
    f1_dict = {names[int(cls)]: 0 for cls in unique_classes}
    # Calculate TP, FP, FN, precision, recall, and F1-score for each class
    for cls in unique_classes:
        correct = filtered_stats[0]
        pred_cls = filtered_stats[2]
        tcls = filtered_stats[3]

        # Filter to only include instances of the current class
        correct_cls = correct[pred_cls == cls]
        tcls_cls = tcls[tcls == cls]

        # Calculate TP and FP
        TP_dict[names[int(cls)]] += correct_cls.sum().item()
        FP_dict[names[int(cls)]] += (~correct_cls).sum().item()

        # Calculate FN
        FN_dict[names[int(cls)]] += len(tcls_cls) - correct_cls.sum().item()

        # Calculate precision, recall, and F1-score
        print("TP: ", TP_dict[names[int(cls)]])
        print("FP: ", FP_dict[names[int(cls)]])
        print("FN: ", FN_dict[names[int(cls)]])

        # Calculate precision, recall, and F1-score with division by zero handling
        if TP_dict[names[int(cls)]] + FP_dict[names[int(cls)]] == 0:
            precision_dict[names[int(cls)]] = 0.0
        else:
            precision_dict[names[int(cls)]] = TP_dict[names[int(cls)]] / (TP_dict[names[int(cls)]] + FP_dict[names[int(cls)]])

        if TP_dict[names[int(cls)]] + FN_dict[names[int(cls)]] == 0:
            recall_dict[names[int(cls)]] = 0.0
        else:
            recall_dict[names[int(cls)]] = TP_dict[names[int(cls)]] / (TP_dict[names[int(cls)]] + FN_dict[names[int(cls)]])

        if precision_dict[names[int(cls)]] + recall_dict[names[int(cls)]] == 0:
            f1_dict[names[int(cls)]] = 0.0
        else:
            f1_dict[names[int(cls)]] = 2 * (precision_dict[names[int(cls)]] * recall_dict[names[int(cls)]]) / (
                    precision_dict[names[int(cls)]] + recall_dict[names[int(cls)]])

    # Calculate sums and overall precision, recall, and F1-score
    sum_TP = sum(TP_dict.values())
    sum_FP = sum(FP_dict.values())
    sum_FN = sum(FN_dict.values())

    # Calculate overall precision, recall, and F1-score with division by zero handling
    sum_precision = sum_TP / (sum_TP + sum_FP) if sum_TP + sum_FP > 0 else 0.0
    sum_recall = sum_TP / (sum_TP + sum_FN) if sum_TP + sum_FN > 0 else 0.0
    sum_f1 = 2 * (sum_precision * sum_recall) / (
            sum_precision + sum_recall) if sum_precision + sum_recall > 0 else 0.0

    # Add the sums to the dictionaries
    TP_dict["sum"] = sum_TP
    FP_dict["sum"] = sum_FP
    FN_dict["sum"] = sum_FN
    precision_dict["sum"] = sum_precision
    recall_dict["sum"] = sum_recall
    f1_dict["sum"] = sum_f1

    # Print the precision, recall, and F1-score for each class
    for cls in unique_classes:
        print(
            f"Class {names[int(cls)]}: Precision = {precision_dict[names[int(cls)]]}, Recall = {recall_dict[names[int(cls)]]}, F1-score = {f1_dict[names[int(cls)]]}")
    print(
        f"Overall: Precision = {precision_dict['sum']}, Recall = {recall_dict['sum']}, F1-score = {f1_dict['sum']}")

    # Save the results to a JSON file
    results_dict = {
        "TP": TP_dict,
        "FP": FP_dict,
        "FN": FN_dict,
        "precision": precision_dict,
        "recall": recall_dict,
        "f1": f1_dict
    }

    # Round all values in the results_dict to 2 decimal places
    for key in results_dict:
        for subkey in results_dict[key]:
            if key == "TP" or key == "FP" or key == "FN":
                results_dict[key][subkey] = int(results_dict[key][subkey])
            else:
                results_dict[key][subkey] = round(results_dict[key][subkey]*100, 2)

    with open(os.path.join(save_dir, 'results.json'), 'w') as json_file:
        json.dump(results_dict, json_file)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or nc < 50) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    #t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    #print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    plots = True
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=names_new)

    # Return results
    print(f"Results saved to {save_dir}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(data_label_files)).tolist()), maps


def parse_xml(xml_path, cls_id_dict={"motorcycle": 0, "car": 1, "truck": 2, "bus": 3, "person": 4, "bicycle": 5,
                                     "e-scooter": 6, "pedestrian": 4}):
    folder_path = os.path.dirname(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract relevant information from the XML
    filename = root.find('filename').text
    image_path = os.path.join(folder_path, filename)
    label_objects = root.findall('object')

    # Get height and width of image
    height = int(root.find('size').find('height').text)
    width = int(root.find('size').find('width').text)

    # Process labels
    cls_labels = []
    box_labels = []
    score_labels = []
    for obj in label_objects:
        cls_label = obj.find('name').text
        ymin = int(float(obj.find('bndbox').find('ymin').text))
        xmin = int(float(obj.find('bndbox').find('xmin').text))
        ymax = int(float(obj.find('bndbox').find('ymax').text))
        xmax = int(float(obj.find('bndbox').find('xmax').text))
        score = float(obj.find('score').text)

        box_label = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
        cls_labels.append(cls_label)
        box_labels.append(box_label)
        score_labels.append(score)

    labels = []
    boxes = []
    scores = []
    for cls_label, box_label, score_label in zip(cls_labels, box_labels, score_labels):
        labels.append(cls_id_dict[str(cls_label)])
        boxes.append([box_label["xmin"], box_label["ymin"], box_label["xmax"], box_label["ymax"]])
        scores.append(score_label)

    return image_path, labels, boxes, height, width, scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='pseudo_label_evaluator.py')
    parser.add_argument('--pseudo_label_path', nargs='+', type=str,
                        default="",
                        help='Pseudo label path')
    parser.add_argument('--data', type=str,
                        default="",
                        help='*.data path')
    parser.add_argument('--task', default='test', help='train, val, test, speed or study')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--project',
                        default="",
                        help='save to project/name')
    parser.add_argument('--name', default='', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--extra-conf-thres', type=float, default=0.01, help='object confidence threshold')
    parser.add_argument('--ignore_files_without_detection', type=bool, default=True, help='ignore files without detection')

    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file

    if opt.task in ('train', 'val', 'test'):  # run normally
        test_pseudo_labels(opt.data,
                   opt.pseudo_label_path,
                   opt.extra_conf_thres,
                   opt.ignore_files_without_detection,
                   opt.verbose
                   )
    else:
        # Raise not implemented error
        raise NotImplementedError("Task is not implemented")

