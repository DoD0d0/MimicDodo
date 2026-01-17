from pathlib import Path

# crude but effective
EXAMPLE_ROBOT_DATA_PATH = Path.home() / ".example_robot_data"

def getModelPath(subpath=""):
    return str(EXAMPLE_ROBOT_DATA_PATH / subpath)
