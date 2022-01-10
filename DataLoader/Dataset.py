import os
from os.path import join
from DatapointHandler import DatapointHandler
from Datapoint import Datapoint

# Todo: Add documentation, make DataPoint more readable, rename to Dataset (without cap letters)
class Dataset:
    """
    Class for dataset parsing and loading into Python
    """
    def __init__(self, path) -> None:
        """
        :param path: The dataset root directory
        """
        all_dirs = sorted(next(os.walk(path))[1])
        data_dirs = [dir for dir in all_dirs if DatapointHandler.check_structure(join(path, dir))]
        self._data = [DatapointHandler(path=join(path, dir)) for dir in data_dirs]

    def __getitem__(self, item) -> Datapoint:
        """
        :param item: The index of the datapoint in the dataset
        :return: A DataPoint object
        """
        return self._data[item].dp

    def __len__(self) -> int:
        return len(self._data)
