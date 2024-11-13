"""
Created on Oct 16 12:36

@author: ISAC - pettirsch
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

from PIL import Image, ImageDraw

# Create class YOLOv7DatasetTester
class YOLOv7DatasetTester:
    # Constructor
    def __init__(self, dataset_path = None, splits = ["train", "val", "test"], images_per_split = 3,
                 class_id_to_name_mapping = {0: "motorcycle", 1: "car", 2: "truck", 3: "bus", 4: "person",
                                                5: "bicycle", 6: "e-scooter"}):
        self._dataset_path = dataset_path
        self._splits = splits
        self._images_per_split = images_per_split

    def test_dataset(self):
        for split in self._splits:
            print("Split: {}".format(split))
            self._test_split(split)

    def _test_split(self, split):
        # Get all annotations
        annotations = self._get_annotations(split)

        # select self._images_per_split random annotations
        random_annotations = random.sample(annotations, self._images_per_split)
        for annotation in random_annotations:
            self._test_annotation(annotation)

    def _get_annotations(self, split):
        annotation_file_list = []
        for file in os.listdir(os.path.join(self._dataset_path, "labels", split)):
            if file.endswith(".txt"):
                annotation_file_list.append(os.path.join(self._dataset_path,"labels",split, file))
        return annotation_file_list

    def _test_annotation(self, annotation):
        with open(annotation, "r") as file:
            annotation_list = file.read().split("\n")[:-1]
            annotation_list = [x.split(" ") for x in annotation_list]
            annotation_list = [[float(y) for y in x] for x in annotation_list]

        # Get the corresponding image file
        image_file = annotation.replace("labels", "images").replace("txt", "png")
        image_file = str(image_file)
        try:
            assert os.path.exists(image_file)
        except:
            print("test")

        # Load the image
        image = Image.open(image_file)

        # Plot the Bounding Box
        self._plot_bounding_box(image, annotation_list)


    def _plot_bounding_box(self,image, annotation_list):
        annotations = np.array(annotation_list)
        w, h = image.size

        plotted_image = ImageDraw.Draw(image)

        transformed_annotations = np.copy(annotations)
        transformed_annotations[:, [1, 3]] = annotations[:, [1, 3]] * w
        transformed_annotations[:, [2, 4]] = annotations[:, [2, 4]] * h

        transformed_annotations[:, 1] = transformed_annotations[:, 1] - (transformed_annotations[:, 3] / 2)
        transformed_annotations[:, 2] = transformed_annotations[:, 2] - (transformed_annotations[:, 4] / 2)
        transformed_annotations[:, 3] = transformed_annotations[:, 1] + transformed_annotations[:, 3]
        transformed_annotations[:, 4] = transformed_annotations[:, 2] + transformed_annotations[:, 4]

        for ann in transformed_annotations:
            obj_cls, x0, y0, x1, y1 = ann
            plotted_image.rectangle(((x0, y0), (x1, y1)))

            plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])

        plt.imshow(np.array(image))
        plt.show()

        # Wait for key press
        cv2.waitKey(0)

if __name__ == "__main__":
    # Parameter
    dataset_path = "/data/Arnd/paper_weakly_supervised_viewpoint_adaption/training_data/AutomotiveThermo/x_t-B1_test_aut_therm"
    splits = ["test"]
    images_per_split = 10
    class_id_to_name_mapping = {0: "motorcycle", 1: "car", 2: "truck", 3: "bus", 4: "person",
                                5: "bicycle", 6: "e-scooter"}

    # Create YOLOv7DatasetTester object
    tester = YOLOv7DatasetTester(dataset_path, splits, images_per_split, class_id_to_name_mapping)
    tester.test_dataset()