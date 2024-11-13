"""
Created on Oct 16 10:10

@author: ISAC - pettirsch
"""

import os
import xml.etree.ElementTree as ET


def check_folder(folder, splitDict):
    data_folder_list_train = []
    data_folder_list_val = []
    data_folder_list_test = []
    sub_folder_list = []
    banned_idx_train = []
    banned_idx_val = []
    banned_idx_test = []
    for root, dirs, files in os.walk(folder):
        for dir in dirs:
            path = os.path.join(root, dir)
            if ".zip" in path or "Zip" in path:
                continue
            if "vor" in path or "Vor" in path:
                continue
            if dir in splitDict["train"]:
                idx = splitDict["train"].index(dir)
                if idx in banned_idx_train:
                    continue
                banned_idx_train.append(idx)
                data_folder_list_train.append(os.path.join(root, dir))
            elif dir in splitDict["val"]:
                idx = splitDict["val"].index(dir)
                if idx in banned_idx_val:
                    continue
                banned_idx_val.append(idx)
                data_folder_list_val.append(os.path.join(root, dir))
            elif dir in splitDict["test"]:
                idx = splitDict["test"].index(dir)
                if idx in banned_idx_test:
                    continue
                banned_idx_test.append(idx)
                data_folder_list_test.append(os.path.join(root, dir))
            else:
                print("Subfolder {} not in split dict".format(dir))


    return data_folder_list_train, data_folder_list_val, data_folder_list_test, sub_folder_list


# Create class YOLOv7DatasetCreator
class YOLOv7DatasetCreator:
    # Constructor
    def __init__(self, split="train",
                 xml_dataset_path=None, target_path=None,
                 class_name_to_id_mapping={"motorcycle": 0, "car": 1, "truck": 2, "bus": 3, "person": 4,
                                           "bicycle": 5, "e-scooter": 6}):

        self._split = split
        self._class_name_to_id_mapping = class_name_to_id_mapping
        self._xml_dataset_path = xml_dataset_path
        self._target_path = target_path

        # Create target path
        self._create_target_path()

    # Create the dataset
    def create_dataset(self):

        # Create the dataset
        xml_files = self._get_all_xml_files_in_folders_and_subfolders([self._xml_dataset_path])

        # For each xml file
        for xml_file in xml_files:
            print("Split: {}. Processing file {} of {}".format(self._split, xml_files.index(xml_file), len(xml_files)),
                  end="\r")

            # Convert xml to yolo format
            try:
                self._convert_xml_to_yolov7(xml_file)
            except:
                print("Error while processing file {}".format(xml_file))

        txt_files = self._get_all_txt_files_in_folders_and_subfolders([self._xml_dataset_path])
        for txt_file in txt_files:
            print("Split: {}. Processing file {} of {}".format(self._split, txt_files.index(txt_file), len(txt_files)),
                  end="\r")
            try:
                self._copy_txt_to_yolov7(txt_file)
            except:
                print("Error while processing file {}".format(txt_file))

    def _get_xml_files(self, data_folder_list):
        xml_files = []
        for data_folder in data_folder_list:
            for file in os.listdir(data_folder):
                if file.endswith(".xml"):
                    xml_files.append(os.path.join(data_folder, file))
        return xml_files

    def _get_all_xml_files_in_folders_and_subfolders(self, data_folder_list):
        xml_files = []
        for data_folder in data_folder_list:
            for root, dirs, files in os.walk(data_folder):
                for file in files:
                    if file.endswith(".xml"):
                        xml_files.append(os.path.join(root, file))
        return xml_files

    def _get_all_txt_files_in_folders_and_subfolders(self, data_folder_list):
        txt_files = []
        for data_folder in data_folder_list:
            for root, dirs, files in os.walk(data_folder):
                for file in files:
                    if file.endswith(".txt"):
                        txt_files.append(os.path.join(root, file))
        return txt_files

    # Create target path
    def _create_target_path(self):
        if not os.path.exists(self._target_path):
            os.makedirs(self._target_path)

        if not os.path.exists(os.path.join(self._target_path, "labels")):
            os.makedirs(os.path.join(self._target_path, "labels"))

        if not os.path.exists(os.path.join(self._target_path, "labels", self._split)):
            os.makedirs(os.path.join(self._target_path, "labels", self._split))

        if not os.path.exists(os.path.join(self._target_path, "images")):
            os.makedirs(os.path.join(self._target_path, "images"))

        if not os.path.exists(os.path.join(self._target_path, "images", self._split)):
            os.makedirs(os.path.join(self._target_path, "images", self._split))

    # Convert the info dict to the required yolo format and write it to disk
    def _convert_xml_to_yolov7(self, xml_path):

        # First get labels and boxes from xml
        try:
            image_path, labels, boxes, image_h, image_w  = self._parse_xml(xml_path)
        except:
            print("Error while parsing xml file: {}".format(xml_path))
            raise ValueError

        print_buffer = []

        # For each object
        for i, label in enumerate(labels):
            try:
                class_id = class_name_to_id_mapping[label]
            except KeyError:
                print("Invalid Class: {}. Must be one from {}".format(label, class_name_to_id_mapping.keys()))
                raise KeyError

            # Get bbox co-ordinates
            b = boxes[i]

            # Transform the bbox co-ordinates as per the format required by YOLO v7
            b_center_x = (b["xmin"] + b["xmax"]) / 2
            b_center_y = (b["ymin"] + b["ymax"]) / 2
            b_width = (b["xmax"] - b["xmin"])
            b_height = (b["ymax"] - b["ymin"])

            # Normalise the co-ordinates by the dimensions of the image
            b_center_x /= image_w
            b_center_y /= image_h
            b_width /= image_w
            b_height /= image_h

            # Write the bbox details to the file
            print_buffer.append(
                "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))

        # Get Filename from image path
        folder_name = os.path.basename(os.path.dirname(image_path))
        filename = os.path.basename(image_path)
        #filename = folder_name + "_" + os.path.basename(image_path).split("_")[-1]

        # Name of the file which we have to save
        save_file_name = os.path.join(self._target_path,"labels", self._split,filename.replace("png", "txt"))
        save_img_name = os.path.join(self._target_path, "images", self._split,filename)

        # Save the annotation to disk
        print("\n".join(print_buffer), file=open(save_file_name, "w"))

        # Save the image to disk
        if ".mp4" in image_path:
            print("stop")
        os.system("cp {} {}".format(image_path, save_img_name))

    def _copy_txt_to_yolov7(self, txt_path):

        image_path = txt_path.replace("labels", "images").replace(".txt", ".png")
        save_img_name = os.path.join(self._target_path, "images", self._split, os.path.basename(image_path))
        save_file_name = os.path.join(self._target_path, "labels", self._split, os.path.basename(txt_path))

        os.system("cp {} {}".format(image_path, save_img_name))
        os.system("cp {} {}".format(txt_path, save_file_name))


    # Function to handle xml
    def _parse_xml(self, xml_path):
        folder_path = os.path.dirname(xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract relevant information from the XML
        filename = root.find('filename').text
        filename = filename.replace(".mp4", "")
        image_path = os.path.join(folder_path, filename)
        label_objects = root.findall('object')

        # Get height and width of image
        height = int(root.find('size').find('height').text)
        width = int(root.find('size').find('width').text)

        # Process labels
        cls_labels = []
        box_labels = []
        for obj in label_objects:
            cls_label = obj.find('name').text
            ymin = int(float(obj.find('bndbox').find('ymin').text))
            xmin = int(float(obj.find('bndbox').find('xmin').text))
            ymax = int(float(obj.find('bndbox').find('ymax').text))
            xmax = int(float(obj.find('bndbox').find('xmax').text))

            box_label = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
            cls_labels.append(cls_label)
            box_labels.append(box_label)

        labels = []
        boxes = []
        for cls_label, box_label in zip(cls_labels, box_labels):
            labels.append(str(cls_label))
            boxes.append(box_label)

        return image_path, labels, boxes, height, width

if __name__ == '__main__':

    dataset_folder = "" # Folder to specific dataset containing Train, Val and Test folder with images and csv files

    curr_splits = ["Train", "Val", "Test"] # Current split folder
    target_splits = ["train", "val", "test"] # Define the splits
    target_path = ""

    class_name_to_id_mapping = {"Motorrad": 0, "motorcycle": 0, "motorcylce":0,
                                "car": 1, "PKW": 1,
                                "truck": 2, "lkw":2, "LKW": 2, "truck´": 2,
                                "bus": 3, "Bus": 3,
                                "person": 4, "Fussgaenger": 4, "pedestrian": 4,
                                "bicycle": 5, "bicylce": 5, "Fahrradfahrer": 5, "Fahrrad": 5,
                                "e-scooter": 6,"E-Scooter":6, "Rollerfahrer": 6}


    for split_idx, curr_split in enumerate(curr_splits):

        data_path = os.path.join(dataset_folder, curr_split)
        split_target_path = os.path.join(target_path, target_splits[split_idx])

        # Create Dataset
        dataset_creator = YOLOv7DatasetCreator(split=target_splits[split_idx],
                                               xml_dataset_path=data_path,
                                               target_path=split_target_path,
                                               class_name_to_id_mapping=class_name_to_id_mapping)
        # Create Dataset
        dataset_creator.create_dataset()



