import os
import shutil

from MFCC import MFCC
from DataLoader import DataLoader


def load_data(cepstra_load_dir=None, transcript_load_dir=None):

    if cepstra_load_dir:
        cepstra, cepstra_paths = MFCC.load_cepstra(cepstra_load_dir)
    else:
        cepstra, cepstra_paths = (None, None)

    if transcript_load_dir:
        labels, label_paths = DataLoader.load_labels(transcript_load_dir)
    else:
        labels, label_paths = (None, None)

    if cepstra and labels:
        return cepstra, cepstra_paths, labels, label_paths
    elif cepstra and not labels:
        return cepstra, cepstra_paths
    elif not cepstra and labels:
        return labels, label_paths
    else:
        print("No data was loaded.")
        return None
