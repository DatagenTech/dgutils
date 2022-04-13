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
        camera_folder_path = join(path, 'environment_?????', 'camera*')
        visible_spectrum_paths = glob(join(camera_folder_path, 'visible_spectrum*.png'))
        ir_paths = glob(join(camera_folder_path, 'infrared_spectrum*.png'))
        render_paths = visible_spectrum_paths + ir_paths
        render_paths.sort()
        render_paths = filter(DatapointHandler.check_structure, render_paths)
        self._data = [DatapointHandler(render_path=path) for path in render_paths]

    def __getitem__(self, item) -> Datapoint:
        """
        :param item: The index of the datapoint in the dataset
        :return: A DataPoint object
        """
        return self._data[item].dp

    def __len__(self) -> int:
        return len(self._data)
