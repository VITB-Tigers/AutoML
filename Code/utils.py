# General utility functions for the project

import yaml
import pandas as pd

def load_config():
    """
    Load the configuration file
    """
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        return config


def read_file(file_path):
    """
    Read a file
    """
    try :
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None