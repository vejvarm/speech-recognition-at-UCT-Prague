# helper functions for this project
import logging
import random
import json

import numpy as np
import soundfile as sf  # for loading audio in various formats (OGG, WAV, FLAC, ...)

from tensorflow.python.client import device_lib


def console_logger(name=__name__, level=logging.WARNING):

    if not isinstance(level, (str, int)):
        raise TypeError("Logging level data type is not recognised. Should be str or int.")

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(level)
    formatter = logging.Formatter('%(levelname)7s (%(name)s) %(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def load_config(config):
    if type(config) is dict:
        return config
    elif type(config) is str:
        try:
            with open(config, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print("The given string doesn't lead to a valid json file path.")
            return False
    else:
        print("The config file is not a dictionary or a string with path to json file.")
        return False


def check_equal(iterator):
    """check if all elements in an iterator are equal

    :return True if all elements in an iterator are equal (==)
    """
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def random_shuffle(iter1, iter2, seed=0):
    """shuffle two iterators the same way

    :return tuple of the two lists shuffled the same way
    """
    list_zip = list(zip(iter1, iter2))
    random.seed(seed)
    random.shuffle(list_zip)

    return zip(*list_zip)


def list_to_padded_array(data_list, max_length=None):
    """convert input list of data into consistent shape numpy array (padded with zeros to max_length)"""
    if not max_length:
        max_length = max(elem.shape[0] for elem in data_list)
    return np.array([np.pad(elem, ((0, max_length - elem.shape[0]), *((0, 0), )*(elem.ndim-1)), mode="constant")
                     for elem in data_list])


def get_available_devices():
    """get all the cpu and gpu device names available on the current system

    :return dict with "gpu" and "cpu" keys containing lists of devices with given type
    """
    device_types = ["gpu", "cpu"]
    devices = {"gpu": [], "cpu": []}
    local_device_protos = device_lib.list_local_devices()

    for dev_type in device_types:
        devices[dev_type] = [x.name for x in local_device_protos if x.device_type == dev_type.upper()]

    return devices


def extract_channel(signal, channel_number):
    """Extract single channel from a multi-channel (stereo) audio signal"""
    try:
        return signal[:, channel_number]
    except IndexError:
        return signal


# TODO: load directly from microphone
def load_speech(audiofile):
    """ load audiofile from disk

    :param audiofile: (str) path to audiofile
    :return: (Tuple[ndarray[float64], int]) numpy array with audiosignal and its sample rate
    """
    signal, fs = sf.read(audiofile)
    signal = extract_channel(signal, 0)  # convert signal from stereo to mono by extracting channel 0

    return signal, fs

