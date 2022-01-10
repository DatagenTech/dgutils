import copy
import os.path

import numpy as np

from Datapoint import Datapoint
from DatapointHandler import DatapointHandler


def save_tmpl_datapoint(dp: Datapoint):
    """
    Saves a Datapoint as a template file.
    Be careful! Overwrites any existing template file
    :param dp: The datapoint to be saved
    """
    if os.path.exists(DatapointHandler.DP_STRUCT_FILE):
        raise RuntimeError("Template file already exists. Please remove it before saving a new one.")
    with open(DatapointHandler.DP_STRUCT_FILE, 'w') as file:
        file.write(dp.repr_aux(struct_only=True))


# Some tests for the validator
# TODO: Add more tests for the validator
def validator_unit_test():
    with open(DatapointHandler.DP_STRUCT_FILE, 'r') as file:
        dp_tmpl = file.read()

    dp = DatapointHandler(path='./Data/environment_00002').dp
    assert dp.repr_aux(struct_only=True) == dp_tmpl

    dp_dict_tampered = copy.deepcopy(dp.__dict__)
    # Purposedly adding a new dimension to the keypoints array
    dp_dict_tampered['keypoints_2d_coordinates'] = dp_dict_tampered['keypoints_2d_coordinates'][np.newaxis]
    dp_tampered = Datapoint(**dp_dict_tampered)
    assert dp_tampered.repr_aux(struct_only=True) != dp_tmpl

    dp_dict_tampered = copy.deepcopy(dp.__dict__)
    dp_dict_tampered['facial_hair_included'] = not dp_dict_tampered['facial_hair_included']
    dp_tampered = Datapoint(**dp_dict_tampered)
    assert dp_tampered.repr_aux(struct_only=True) == dp_tmpl

