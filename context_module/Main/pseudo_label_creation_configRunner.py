"""
Created on Mar 06 2024 08:57

@author: ISAC - pettirsch
"""

import os
import subprocess
import argparse

super_dir = ""

def main(config_files):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))


    for config_file in config_files:
        print("Running config file: ", config_file)

        # get filename without path
        config_file_name = os.path.basename(config_file)

        # remove .json from filename
        config_file_name = config_file_name.replace(".json", "")

        # make dir /data/Arnd/paper_weakly_supervised_viewpoint_adaption/experiments/stage_2/config_file_name
        os.makedirs(os.path.join(super_dir, config_file_name), exist_ok=True)
        os.makedirs(os.path.join(super_dir, config_file_name, "labels"), exist_ok=True)
        os.makedirs(os.path.join(super_dir, config_file_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(super_dir, config_file_name, "raw_images"), exist_ok=True)
        os.makedirs(os.path.join(super_dir, config_file_name, "stats"), exist_ok=True)

        # Form the path to the main script
        main_script_path = os.path.join(script_dir, "pseudo_label_creation.py")

        # Form the command to execute the main script with the current configuration file
        command = ["python", main_script_path, "--configPath", config_file]

        # Execute the command
        subprocess.run(command)

if __name__ == "__main__":

    # Example invoke: python config_runner.py config1.json config2.json config3.json

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run main script with multiple configuration files")
    parser.add_argument("config_files", nargs="+", help="List of configuration files")
    args = parser.parse_args()

    # Call the main function with the provided configuration files
    main(args.config_files)