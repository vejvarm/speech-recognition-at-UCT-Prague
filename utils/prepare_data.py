import os

from itertools import compress

import numpy as np

from MFCC import MFCC
from DataLoader import PDTSCLoader


def extract_filenames(audio_folder, transcript_folder):
    audio_files = [os.path.splitext(f) for f in os.listdir(audio_folder)
                   if os.path.isfile(os.path.join(audio_folder, f))]
    transcript_files = [os.path.splitext(f) for f in os.listdir(transcript_folder)
                        if os.path.isfile(os.path.join(transcript_folder, f))]

    files = []
    for file1, file2 in zip(audio_files, transcript_files):
        err_message = "{} =/= {}".format(file1[0], file2[0])
        assert file1[0] == file2[0], err_message
        files.append((audio_folder+file1[0]+file1[1], transcript_folder+file2[0]+file2[1]))

    return files


def prepare_data(files, save_folder, filter_nan=True):
    for file in files:
        pdtsc = PDTSCLoader([file[0]], [file[1]])
        labels = pdtsc.transcripts_to_labels()  # list of lists of 1D numpy arrays
        audio, fs = pdtsc.load_audio()

        mfcc = MFCC(audio[0], fs)  # TODO: make MFCC work for more audiofiles
        cepstra = mfcc.transform_data()  # list of 2D arrays

        #    mfcc.plot_cepstra(cepstra, figstart=1, nplots=1)

        # filter out cepstra which are containing nan values
        if filter_nan:
            # boolean list where False marks cepstra in which there is at least one nan value present
            mask_nan = [not np.isnan(cepstrum).any() for cepstrum in cepstra]

            # mask out cepstra and their corresponding labels with nan values
            cepstra = list(compress(cepstra, mask_nan))
            labels[0] = list(compress(labels[0], mask_nan))


        # SAVE Cepstra to files (features)
        subfolder = os.path.splitext(os.path.split(file[0])[1])[0]
        mfcc.save_cepstra(cepstra, save_folder + subfolder, exist_ok=True)
        print(file[0] + ' transformed and saved into {}.'.format(os.path.abspath(save_folder) + subfolder))

        # SAVE Transcripts to files (labels)
        pdtsc.save_labels(labels, save_folder, exist_ok=True)
        print(file[1] + ' transformed and saved into {}.'.format(os.path.abspath(save_folder) + subfolder))


if __name__ == '__main__':
    # extracting audiofiles, transforming into cepstra and saving to separate folders
    audio_folder = "../data/raw/audio/"
    transcript_folder = "../data/raw/transcripts/"
    save_folder = '../data/train/'

    files = extract_filenames(audio_folder, transcript_folder)

    prepare_data(files, save_folder)
