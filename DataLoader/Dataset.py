from os.path import join
from glob import glob

from DatapointHandler import DatapointHandler
from Datapoint import Datapoint


class Dataset:
    """
    Class for dataset parsing and loading into Python
    """
    def __init__(self, path) -> None:
        """
        :param path: The dataset root directory
        """
        render_paths = glob(join(path, 'environment_?????', 'camera*', 'visible_spectrum*.png'))
        render_paths = filter(DatapointHandler.check_structure, render_paths)
        self._data = [DatapointHandler(render_path=path, check_integrity=True) for path in render_paths]

    def __getitem__(self, item) -> Datapoint:
        """
        :param item: The index of the datapoint in the dataset
        :return: A DataPoint object
        """
        return self._data[item].dp

    def __len__(self) -> int:
        return len(self._data)
