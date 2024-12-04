# General utility functions for the project

import yaml
import pandas as pd

# Load the configuration file
def load_config():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        return config

# Read a file and return the data
def read_file(file_path):
    try :
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None
