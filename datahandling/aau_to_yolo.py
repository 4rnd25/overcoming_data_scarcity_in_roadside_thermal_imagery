"""
Created on Jun 20 06:51

@author: ISAC - pettirsch
"""

import json
import os

class_name_to_id_mapping = {"person": 4, "bicycle": 5,
                            "car": 1, "motorbike": 0, "bus": 3, "truck": 2}

def create_aau_yolo_dataset(input_path=None, output_path=None):
    train_file_list = []
    val_file_list = []
    test_file_list = []

    # First read in annotations
    ann_file = os.path.join(input_path, 'aauRainSnow-thermal.json')
    with open(ann_file, 'r') as file:
        raw_data = json.load(file)

    # SEcon create target paths
    for split in ['train', 'val', 'test']:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if not os.path.exists(os.path.join(output_path, "labels")):
            os.makedirs(os.path.join(output_path, "labels"))

        if not os.path.exists(os.path.join(output_path, "labels", split)):
            os.makedirs(os.path.join(output_path, "labels", split))

        if not os.path.exists(os.path.join(output_path, "images")):
            os.makedirs(os.path.join(output_path, "images"))

        if not os.path.exists(os.path.join(output_path, "images", split)):
            os.makedirs(os.path.join(output_path, "images", split))


    # Analyse data
    data_anlaysis = {"Egensevej": {"num_images": 0, "choosen_train": 0, "choosen_val": 0, "choosen_test": 0},
                     "Hadsundvej": {"num_images": 0, "choosen_train": 0, "choosen_val": 0, "choosen_test": 0},
                     "Hasserisvej": {"num_images": 0, "choosen_train": 0, "choosen_val": 0, "choosen_test": 0},
                     "Hjorringvej": {"num_images": 0, "choosen_train": 0, "choosen_val": 0, "choosen_test": 0},
                     "Hobrovej": {"num_images": 0, "choosen_train": 0, "choosen_val": 0, "choosen_test": 0},
                     "Ostre": {"num_images": 0, "choosen_train": 0, "choosen_val": 0, "choosen_test": 0},
                     "Ringvej": {"num_images": 0, "choosen_train": 0, "choosen_val": 0, "choosen_test": 0}}
    for id, image_info in enumerate(raw_data["images"]):
        place = image_info["file_name"].split("/")[0]
        data_anlaysis[place]["num_images"] += 1

    categories = raw_data["categories"]

    # Create dataset
    for ann_id, annotation in enumerate(raw_data["annotations"]):

        print("Processing annotation: ", ann_id + 1, "/", len(raw_data["annotations"]))

        category = categories[annotation["category_id"]-1]["name"]
        if category not in class_name_to_id_mapping:
            continue
        category_id = class_name_to_id_mapping[category]

        c_x = annotation["bbox"][0]
        c_y = annotation["bbox"][1]
        w = annotation["bbox"][2]
        h = annotation["bbox"][3]

        c_x= c_x + w/2
        c_y = c_y + h/2

        if c_x > 640 or c_y > 480 or c_x + w > 640 or c_y + h > 480:
            continue

        if c_x < 0 or c_y < 0 or w < 0 or h < 0:
            continue

        c_x = c_x/640
        c_y = c_y/480
        w = w/640
        h = h/480

        image_info = raw_data["images"][annotation["image_id"]]

        if image_info["id"] != annotation["image_id"]:
            image_info = extensive_image_search(annotation["image_id"], raw_data["images"])

        place = image_info["file_name"].split("/")[0]
        label_file_name = image_info["file_name"].replace("/", "_").replace(".png", ".txt")

        if image_info["file_name"] in train_file_list:
            split = "train"
        elif image_info["file_name"] in val_file_list:
            split = "val"
        elif image_info["file_name"] in test_file_list:
            split = "test"
        else:
            if data_anlaysis[place]["choosen_train"] < data_anlaysis[place]["num_images"] * 0.50:
                split = "train"
                data_anlaysis[place]["choosen_train"] += 1
                train_file_list.append(image_info["file_name"])
            elif data_anlaysis[place]["choosen_val"] < data_anlaysis[place]["num_images"] * 0.25:
                split = "val"
                data_anlaysis[place]["choosen_val"] += 1
                val_file_list.append(image_info["file_name"])
            else:
                split = "test"
                data_anlaysis[place]["choosen_test"] += 1
                test_file_list.append(image_info["file_name"])



            # Copy image and name it image_info["file_name"].replace("/", "_")
            os.system(
                "cp " + os.path.join(input_path, image_info["file_name"]) + " " + os.path.join(output_path, "images",
                                                                                               split,
                                                                                               image_info[
                                                                                                   "file_name"].replace(
                                                                                                   "/", "_")))

            # Save the annotation to disk
            open(os.path.join(output_path, "labels", split, label_file_name), "w").close()

        # Open annotation file and write annotation
        with open(os.path.join(output_path, "labels", split, label_file_name), "r+") as file:
            # Load current annotations from file and append the annotation to the print buffer
            print_buffer = file.read().splitlines()

            print_buffer.append(
                "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(category_id, c_x, c_y, w, h))

            # Move the file pointer to the beginning of the file to write the updated content
            file.seek(0)

            # Write the updated print buffer to the file with proper newlines
            file.write('\n'.join(print_buffer) + '\n')

            # Truncate the file to the new length (in case it was longer before)
            file.truncate()

def extensive_image_search(image_id, image_infos):
    for i, image_info in enumerate(image_infos):
        if image_info["id"] == image_id:
            return image_info
    raise ValueError("Image not found")


if __name__ == '__main__':
    input_path = ""
    target_path = ""

    create_aau_yolo_dataset(input_path, target_path)