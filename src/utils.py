import yaml
import logging
import os
import sys

def load_config(config_path="config.yaml"):
    """
    Load data from .yaml file and chang them into python dictionary.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"ERROR: File not found at: {config_path}!")
        sys.exit(1)

def get_logger(name):
    """
    Creating logger.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.DEBUG) 
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s',datefmt='%H:%M:%S')

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
    return logger

def ensure_dir(path):
    """
    Creating folder if doesn't exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)