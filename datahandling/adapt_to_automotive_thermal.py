"""
Created on Jun 20 11:04

@author: ISAC - pettirsch
"""

import os
from PIL import Image

class_name_to_id_mapping = {"person": 4, "bicycle": 5,
                            "car": 1, "motorbike": 0, "bus": 3, "truck": 2}

id_map = {0: 4, 1: 5, 2: 1, 3: 0, 5: 3, 7:2}

def adapt_labels(label_folder=None):

    for file in os.listdir(label_folder):

        print("Processing file: ", file)

        if not file.endswith(".txt"):
            continue

        with open(os.path.join(label_folder, file), "r") as f:
            lines = f.readlines()

            for i, line in enumerate(lines):

                category_id = int(line.split(" ")[0])

                c_x = (float(line.split(" ")[1]) * (640*(320/640))+ (640-(640*(320/640)))/2)/640
                c_y = (float(line.split(" ")[2]) * (480*(320/640))+ (480-(480*(320/640)))/2)/480
                w = float(line.split(" ")[3]) * (640*(320/640)) /640
                h = float(line.split(" ")[4]) * (480*(320/640))/480

                lines[i] = "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(category_id, c_x, c_y, w, h)


        with open(os.path.join(label_folder, file), "w") as f:
            # Remove all old content
            f.truncate(0)

            # Move the file pointer to the beginning of the file to write the updated content
            f.seek(0)

            # Write the updated print buffer to the file with proper newlines
            f.write('\n'.join(lines) + '\n')

            # Truncate the file to the new length (in case it was longer before)
            f.truncate()


def pad_images(image_folder=None):

    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            file_path = os.path.join(image_folder, filename)
            with Image.open(file_path) as img:
                if img.size == (640, 480):

                    # Resize the image maintaining the aspect ratio to Xx256
                    new_width = 320
                    new_height = int(480 * (new_width / 640))
                    img = img.resize((new_width, new_height), Image.LANCZOS)

                    # Create a new image with the target size and a black (zero) background
                    new_img = Image.new('RGB', (320, 256), (0, 0, 0))

                    # Calculate the position to paste the resized image onto the black background
                    paste_x = (320 - img.width) // 2
                    paste_y = (256 - img.height) // 2

                    # Paste the resized image onto the new image
                    new_img.paste(img, (paste_x, paste_y))

                    #320, 256
                    padded_img = pad_image(new_img)
                    output_path = os.path.join(image_folder, os.path.splitext(filename)[0] + ".png")
                    padded_img.save(output_path, format='png')
                    print(f"Processed and saved: {output_path}")

                    # # Remove the original .bmp file
                    # os.remove(file_path)
                    # print(f"Removed original file: {file_path}")

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
    image_folder = ""
    label_folder = ""

    adapt_labels(label_folder)

    pad_images(image_folder)