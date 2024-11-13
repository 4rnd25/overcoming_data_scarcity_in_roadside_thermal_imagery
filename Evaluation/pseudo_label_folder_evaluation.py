"""
Created on Mar 11 2024 08:35

@author: ISAC - pettirsch
"""

import argparse
import os
import subprocess

parser = argparse.ArgumentParser(prog='pseudo_label_evaluator.py')


def main(folder_path=None, data=None, task="test", project=None, extra_conf_thres=0.01,
         ignore_files_without_detection=True):
    # Iterate over all folder in folder_path
    for folder in os.listdir(folder_path):
        print("Running folder: ", folder)

        if not "x_t-B2" in folder:
            continue

        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Form the path to the main script
        main_script_path = os.path.join(script_dir, "pseudo_label_evaluator.py")

        pseudo_label_path = os.path.join(folder, "labels")
        pseudo_label_path = os.path.join(folder_path, pseudo_label_path)

        # Get last path of folder
        folder_name = os.path.basename(os.path.normpath(folder))
        if ignore_files_without_detection:
            name = folder_name + "_only_w_det"
        else:
            name = folder_name + "_all"

        # Form the command to execute the main script with the current configuration file
        if name in os.listdir(project):
            print("Folder {} already exists".format(folder))
            continue
        else:
            print("Folder {} does not exist".format(folder))

        command = ["python",
                   main_script_path,
                   "--pseudo_label_path",
                   str(pseudo_label_path),
                   "--data",
                   str(data),
                   "--task",
                   str(task),
                   "--project",
                   str(project),
                   "--extra-conf-thres",
                   str(extra_conf_thres),
                   "--name",
                   name,
                   "--ignore_files_without_detection",
                   str(ignore_files_without_detection)]

        # Execute the command
        subprocess.run(command)


if __name__ == "__main__":
    # Example invoke: python config_runner.py config1.json config2.json config3.json

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run main script with folder path")
    parser.add_argument("--folder_path", default="", help="Main Folder path")
    parser.add_argument('--data', type=str,
                        default="",
                        help='*.data path')
    parser.add_argument('--task', default='test', help='train, val, test, speed or study')
    parser.add_argument('--project',
                        default="",
                        help='save to project/name')
    parser.add_argument('--extra_conf_thresh', type=float, default=0.0001, help='object confidence threshold')
    parser.add_argument('--ignore_files_without_detection', type=bool, default=True,
                        help='ignore files without detection')

    args = parser.parse_args()

    # Call the main function with the provided configuration files
    main(args.folder_path, args.data, args.task, args.project, args.extra_conf_thresh,
         args.ignore_files_without_detection)
