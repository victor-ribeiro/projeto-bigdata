import json
import os


def get_config():
    fp = os.path.join(
        os.path.dirname(os.path.abspath("__file__")), "utils", ".config.json"
    )
    with open(fp, "r") as file:
        conf = json.load(fp=file)
        return conf
