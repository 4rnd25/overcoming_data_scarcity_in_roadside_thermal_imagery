"""
Created on Jun 20 11:04

@author: ISAC - pettirsch
"""

import json
import os
from PIL import Image

class_name_to_id_mapping = {"person": 4, "bicycle": 5,
                            "car": 1, "motorbike": 0, "bus": 3, "truck": 2}

id_map = {0: 4, 1: 5, 2: 1, 3: 0, 5: 3, 7:2}

def create_new_id_map(label_path=None):

    splits = ["train", "val", "test"]

    for split in splits:
        label_folder = os.path.join(label_path, split)

        for file in os.listdir(label_folder):

            print("Processing file: ", file)

            if not file.endswith(".txt"):
                continue

            with open(os.path.join(label_folder, file), "r") as f:
                lines = f.readlines()

                idx_to_remove = []
                for i, line in enumerate(lines):
                    category_id_old = int(line.split(" ")[0])
                    if category_id_old in id_map.keys():
                        cetegory_id_new = id_map[category_id_old]
                    else:
                        idx_to_remove.append(i)

                    c_x = (float(line.split(" ")[1]) * 320 + (640-320)/2) /640
                    c_y = (float(line.split(" ")[2]) * 256 + (480-256)/2) /480
                    w = float(line.split(" ")[3]) * 320 /640
                    h = float(line.split(" ")[4]) * 256 /480

                    lines[i] = "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(cetegory_id_new, c_x, c_y, w, h)

                for i in idx_to_remove:
                    lines.pop(i)

            with open(os.path.join(label_folder, file), "w") as f:
                # Remove all old content
                f.truncate(0)

                # Move the file pointer to the beginning of the file to write the updated content
                f.seek(0)

                # Write the updated print buffer to the file with proper newlines
                f.write('\n'.join(lines) + '\n')

                # Truncate the file to the new length (in case it was longer before)
                f.truncate()


def pad_images(image_path=None):

    splits = ["train", "val", "test"]

    for split in splits:
        image_folder = os.path.join(image_path, split)

        for filename in os.listdir(image_folder):
            if filename.endswith(".bmp") or filename.endswith(".Bmp") or filename.endswith(".jpg"):
                file_path = os.path.join(image_folder, filename)
                with Image.open(file_path) as img:
                    if img.size == (320, 256):
                        padded_img = pad_image(img)
                        output_path = os.path.join(image_folder, os.path.splitext(filename)[0] + ".png")
                        padded_img.save(output_path, format='png')
                        print(f"Processed and saved: {output_path}")

                        # Remove the original .bmp file
                        os.remove(file_path)
                        print(f"Removed original file: {file_path}")

def pad_image(input_image, output_size=(640, 480)):
    width, height = input_image.size
    new_width, new_height = output_size

    # Calculate padding
    pad_width = (new_width - width) // 2
    pad_height = (new_height - height) // 2

    # Create a new image with the desired size and a black (zero) background
    new_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))

    # Paste the original image onto the center of the new image
    new_image.paste(input_image, (pad_width, pad_height))

    return new_image


if __name__ == '__main__':
    label_path = "/labels"
    image_path = "/images"

    create_new_id_map(label_path)

    pad_images(image_path)