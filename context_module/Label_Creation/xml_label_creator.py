"""
Created on Jan 12 15:57

@author: ISAC - pettirsch
"""
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
import os

class XML_Label_Creator:
    def __init__(self, outputPath, add_score=False, label_every_nth_frame=15, output_only_if_detections_exist=False):
        self.output_path = outputPath
        self.add_score = add_score
        self.label_every_nth_frame = label_every_nth_frame
        self.output_only_if_detections_exist = output_only_if_detections_exist

    def create_labels(self, image_path, detections, frame_num = None):
        if frame_num != 0:
            if frame_num is not None and frame_num % self.label_every_nth_frame != 0:
                return

        if self.output_only_if_detections_exist:
            if len(detections["boxes"]) < 1:
                return

        annotation = ET.Element("annotation")

        # Get folder and filename from image_path
        folder_img = image_path.split("/")[-2]
        filename_img = image_path.split("/")[-1]

        # Add elements for image information
        folder = ET.SubElement(annotation, "folder")
        folder.text = folder_img

        filename = ET.SubElement(annotation, "filename")
        filename.text = filename_img

        path = ET.SubElement(annotation, "path")
        path.text = image_path

        source = ET.SubElement(annotation, "source")
        database = ET.SubElement(source, "database")
        database.text = "Unknown"

        size = ET.SubElement(annotation, "size")
        width = ET.SubElement(size, "width")
        width.text = "640"

        height = ET.SubElement(size, "height")
        height.text = "480"

        depth = ET.SubElement(size, "depth")
        depth.text = "3"

        segmented = ET.SubElement(annotation, "segmented")
        segmented.text = "0"

        # Add elements for each object in detections
        for i in range(len(detections["boxes"])):
            obj = ET.SubElement(annotation, "object")

            name = ET.SubElement(obj, "name")
            name.text = detections["classes"][i]

            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"

            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"

            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(float(detections["boxes"][i][0]))

            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(float(detections["boxes"][i][1]))

            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(float(detections["boxes"][i][2]))

            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(float(detections["boxes"][i][3]))

            if self.add_score:
                score = ET.SubElement(obj, "score")
                score.text = str(float(detections["scores"][i]))

        # Convert the XML to a formatted string
        xml_string = self.prettify(annotation)

        # remove _number.png from filename
        filename_split = filename_img.split("_")
        filename_new = ""
        for i in range(len(filename_split)-1):
            filename_new += filename_split[i] + "_"
        filename_img = filename_new[:-1] + "_" + str(frame_num) + ".png"

        # Save as xml file
        xml_file = open(os.path.join(self.output_path, filename_img.replace(".png", ".xml")), "w")
        xml_file.write(xml_string)
        xml_file.close()


    def prettify(self, elem):
        # Return a pretty-printed XML string
        rough_string = ET.tostring(elem, "utf-8")
        reparsed = parseString(rough_string)
        return reparsed.toprettyxml(indent="\t")