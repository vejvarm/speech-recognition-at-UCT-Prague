import os
import shutil

from MFCC import MFCC
from DataLoader import DataLoader


def choose_cepstra_frame_length(cepstra_load_dir, transcript_load_dir, cepstra_save_dir, transcript_save_dir,
                                min_frame_length=50, max_frame_length=3000):
    """

    :param cepstra_load_dir: folder from which to load cepstra
    :param transcript_load_dir: folder from which to load transcripts
    :param cepstra_save_dir: folder to which to save the passing cepstra
    :param transcript_save_dir: folder to which to save the passing transcripts
    :param min_frame_length: minimum time length of cepstrum in frames
    :param max_frame_length: maximum time length of cepstrum in frames
    :return:
    """
    cepstra, cepstra_paths = MFCC.load_cepstra(cepstra_load_dir)
    labels, label_paths = DataLoader.load_labels(transcript_load_dir)

    assert len(cepstra) == len(labels)

    # normalize the savepaths
    cepstra_save_dir = os.path.normpath(cepstra_save_dir)
    transcript_save_dir = os.path.normpath(transcript_save_dir)

    for i in range(len(cepstra)):
        assert len(cepstra[i]) == len(labels[i])
        for cepstrum, cepstrum_path, label_path in zip(cepstra[i], cepstra_paths[i], label_paths[i]):
            if min_frame_length <= cepstrum.shape[0] <= max_frame_length:
                # create save paths for files which passed the max_frame_length test
                cepstrum_save_path = os.path.join(cepstra_save_dir,
                                                  os.path.join(*cepstrum_path.split("\\")[-2:]))
                label_save_path = os.path.join(transcript_save_dir,
                                               os.path.join(*label_path.split("\\")[-2:]))
                # make the save directories
                os.makedirs(os.path.split(cepstrum_save_path)[0], exist_ok=True)
                os.makedirs(os.path.split(label_save_path)[0], exist_ok=True)
                # copy this cepstrum and label to folder of data which passed the max_frame_length test
                shutil.copy2(cepstrum_path, cepstrum_save_path)
                shutil.copy2(label_path, label_save_path)
                print("cepstrum {} and transcript {} passed the test"
                      " (cepstrum shorter than {} frames)".format(os.path.split(cepstrum_path)[1],
                                                                  os.path.split(label_path)[1],
                                                                  max_frame_length)
                      )


if __name__ == '__main__':
    min_frame_length = 30
    max_frame_length = 1000
    load_dir = "../data/train_deltas"
    save_dir = "../data/train_deltas_min_{}_max_{}".format(min_frame_length, max_frame_length)

    choose_cepstra_frame_length(load_dir, load_dir, save_dir, save_dir, min_frame_length, max_frame_length)
