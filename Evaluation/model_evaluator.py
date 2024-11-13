"""
Created on Jan 24 09:59

@author: ISAC - pettirsch
"""

import argparse
import json
import os
from pathlib import Path
from threading import Thread
import cv2

import pdb

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import check_dataset, check_file, check_img_size, \
    box_iou, non_max_suppression, scale_coords, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized, TracedModel


def test_model(data,
               weights=None,
               batch_size=16,
               imgsz=640,
               conf_thres=0.001,
               iou_thres=0.6,  # for NMS
               extra_conf_thres=0.25,
               verbose=False,
               model=None,
               save_dir=Path(''),  # for saving images
               plots=True,
               wandb_logger=None,
               compute_loss=None,
               half_precision=True,
               trace=False):
    # Set logging and device
    set_logging()
    device = select_device(opt.device, batch_size=batch_size)

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size

    if trace:
        model = TracedModel(model, device, imgsz)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)

    # Dataloader
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
    dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=False,
                                   prefix=colorstr(f'{task}: '))[0]

    # Intialize parameters
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    iou_stats = []

    # Initialize variables for TP, FP, FN, TN
    TP, FP, FN, TN = 0, 0, 0, 0

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):

        # use opencv to plot img[0]
        image_np = img[0,:,:,:].permute(1, 2, 0).cpu().numpy()
        image_np = (image_np).astype(np.uint8)

        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out, train_out = model(img, augment=False)  # inference and training outputs
            t0 += time_synchronized() - t

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    if len(tcls)>0:
                        iou_stats.append(torch.zeros(0, 1, dtype=torch.float32))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            iou_batch = torch.zeros(pred.shape[0], 1, dtype=torch.float32, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))


                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv # iou_thres is 1xn
                                iou_batch[pi[j]] = ious[j]
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
            iou_stats.append(iou_batch.cpu())


        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=False, save_dir=save_dir,
                                              names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Compute IoUs
    iou_stats = np.concatenate(iou_stats, 0)
    unique_classes = np.unique(stats[3])
    IoU_list_dict = {names[cls]: [] for cls in unique_classes}
    IoU_list_dict["all"] = []
    IoU_dict = {names[cls]: 0 for cls in unique_classes}
    IoU_dict["all"] = 0
    # Get all TP with IOU > 0.5
    tp = stats[0][stats[0][:, 0] == True]
    pred_cls = stats[2][stats[0][:, 0] == True]
    iou_tp = iou_stats[stats[0][:, 0] == True]

    for cls in unique_classes:
        # Get all TP with cls == pred_cls
        IoU_list = iou_tp[pred_cls == cls]
        IoU_list_dict[names[cls]] = IoU_list
        IoU_list_dict["all"] = np.concatenate((IoU_list_dict["all"], IoU_list[:,0]))
        IoU_dict[names[cls]] = np.mean(IoU_list)

    # Compute extra statistics at fixed score threshold
    conf = stats[1]  # confidence
    filter_inds = np.where(conf > extra_conf_thres)
    correct = stats[0]
    filtered_stats = [stats[0][:, 0][filter_inds], stats[1][filter_inds], stats[2][filter_inds], stats[3]]

    # Get the unique classes
    unique_classes = np.unique(filtered_stats[3])
    print(unique_classes)
    # Initialize dictionaries to store the TP, FP, FN, precision, recall, and F1-score for each class
    TP_dict = {names[cls]: 0 for cls in unique_classes}
    FP_dict = {names[cls]: 0 for cls in unique_classes}
    FN_dict = {names[cls]: 0 for cls in unique_classes}
    precision_dict = {names[cls]: 0 for cls in unique_classes}
    recall_dict = {names[cls]: 0 for cls in unique_classes}
    f1_dict = {names[cls]: 0 for cls in unique_classes}
    # Calculate TP, FP, FN, precision, recall, and F1-score for each class
    for cls in unique_classes:
        # Count only dets with iou > 0.5
        correct = filtered_stats[0]
        pred_cls = filtered_stats[2]
        tcls = filtered_stats[3]

        # Filter to only include instances of the current class
        correct_cls = correct[pred_cls == cls]
        tcls_cls = tcls[tcls == cls]

        # Calculate TP and FP
        TP_dict[names[cls]] += correct_cls.sum().item()
        FP_dict[names[cls]] += (~correct_cls).sum().item()

        # Calculate FN
        FN_dict[names[cls]] += len(tcls_cls) - correct_cls.sum().item()

        # Calculate precision, recall, and F1-score
        print("TP: ", TP_dict[names[cls]])
        print("FP: ", FP_dict[names[cls]])
        print("FN: ", FN_dict[names[cls]])

        # Calculate precision, recall, and F1-score with division by zero handling
        if TP_dict[names[cls]] + FP_dict[names[cls]] == 0:
            precision_dict[names[cls]] = 0.0
        else:
            precision_dict[names[cls]] = TP_dict[names[cls]] / (TP_dict[names[cls]] + FP_dict[names[cls]])

        if TP_dict[names[cls]] + FN_dict[names[cls]] == 0:
            recall_dict[names[cls]] = 0.0
        else:
            recall_dict[names[cls]] = TP_dict[names[cls]] / (TP_dict[names[cls]] + FN_dict[names[cls]])

        if precision_dict[names[cls]] + recall_dict[names[cls]] == 0:
            f1_dict[names[cls]] = 0.0
        else:
            f1_dict[names[cls]] = 2 * (precision_dict[names[cls]] * recall_dict[names[cls]]) / (
                        precision_dict[names[cls]] + recall_dict[names[cls]])

    # Calculate sums and overall precision, recall, and F1-score
    sum_TP = sum(TP_dict.values())
    sum_FP = sum(FP_dict.values())
    sum_FN = sum(FN_dict.values())

    # Calculate overall precision, recall, and F1-score with division by zero handling
    sum_precision = sum_TP / (sum_TP + sum_FP) if sum_TP + sum_FP > 0 else 0.0
    sum_recall = sum_TP / (sum_TP + sum_FN) if sum_TP + sum_FN > 0 else 0.0
    sum_f1 = 2 * (sum_precision * sum_recall) / (sum_precision + sum_recall) if sum_precision + sum_recall > 0 else 0.0

    # Add the sums to the dictionaries
    TP_dict["sum"] = sum_TP
    FP_dict["sum"] = sum_FP
    FN_dict["sum"] = sum_FN
    precision_dict["sum"] = sum_precision
    recall_dict["sum"] = sum_recall
    f1_dict["sum"] = sum_f1

    # Prepare iou dict
    for cls in IoU_dict.keys():
        IoU_dict[cls] = str(round(IoU_dict[cls], 2))
    IoU_dict["sum"] = str(round(np.mean(IoU_list_dict["all"]), 2))

    # Print the precision, recall, and F1-score for each class
    for cls in unique_classes:
        print(
            f"Class {names[cls]}: Precision = {precision_dict[names[cls]]}, Recall = {recall_dict[names[cls]]}, F1-score = {f1_dict[names[cls]]}")
    print(f"Overall: Precision = {precision_dict['sum']}, Recall = {recall_dict['sum']}, F1-score = {f1_dict['sum']}")

    # Save the results to a JSON file
    results_dict = {
        "IoU": IoU_dict,
        "TP": TP_dict,
        "FP": FP_dict,
        "FN": FN_dict,
        "precision": precision_dict,
        "recall": recall_dict,
        "f1": f1_dict
    }

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
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    # Return results
    print(f"Results saved to {save_dir}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='model_evaluator.py')
    parser.add_argument('--weights', nargs='+', type=str,
                        default="",
                        help='model.pt path(s)')
    parser.add_argument('--data', type=str,
                        default="",
                        help='*.data path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--task', default='test', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--project',
                        default="",
                        help='save to project/name')
    parser.add_argument('--name', default='', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--extra-conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--conf-thres', type=float, default=0.0, help='object confidence threshold')

    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    opt.single_cls = False
    print(opt)

    if opt.task in ('train', 'val', 'test'):  # run normally
        test_model(opt.data,
                   opt.weights,
                   opt.batch_size,
                   opt.img_size,
                   opt.conf_thres,
                   opt.iou_thres,
                   opt.extra_conf_thres,
                   opt.verbose
                   )
    else:
        # Raise not implemented error
        raise NotImplementedError("Task is not implemented")
