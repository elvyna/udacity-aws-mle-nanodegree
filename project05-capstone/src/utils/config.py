import os
from ruamel.yaml import YAML


def read_config(config_file_name):
    """
    Retrieve config content. Config is stored as YAML file.
    """
    yaml = YAML()
    config_file_path = os.path.join("config", config_file_name)
    with open(config_file_path) as config_yml:
        config = yaml.load(config_yml)
    return config
