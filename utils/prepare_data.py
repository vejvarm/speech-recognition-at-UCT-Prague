import os

from itertools import compress

import numpy as np

from FeatureExtraction import FeatureExtractor
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


def prepare_data(files, save_folder, feature_type="MFSC", energy=True, deltas=(0, 0), filter_nan=True, sort=True):
    final_cepstra = []
    final_labels = []

    for file in files:
        pdtsc = PDTSCLoader([file[0]], [file[1]])
        labels = pdtsc.transcripts_to_labels()  # list of lists of 1D numpy arrays
        audio, fs = pdtsc.load_audio()

        mfcc = FeatureExtractor(audio[0], fs, feature_type=feature_type, energy=energy, deltas=deltas)
        cepstra = mfcc.transform_data()  # list of 2D arrays

        #    mfcc.plot_cepstra(cepstra, figstart=1, nplots=1)

        # filter out cepstra which are containing nan values
        if filter_nan:
            # boolean list where False marks cepstra in which there is at least one nan value present
            mask_nan = [not np.isnan(cepstrum).any() for cepstrum in cepstra]

            # mask out cepstra and their corresponding labels with nan values
            cepstra = list(compress(cepstra, mask_nan))
            labels[0] = list(compress(labels[0], mask_nan))


        # add cepstra and labels to collective lists
        final_cepstra.extend(cepstra)
        print(file[0].split("/")[-1] + " extracted.")
        final_labels.extend(labels[0])
        print(file[1].split("/")[-1] + " extracted.")

    # sort cepstra and labels by time length (number of frames)
    if sort:
        sort_indices = np.argsort([c.shape[0] for c in final_cepstra])  # indices which sort the lists by cepstra length
        final_cepstra = [final_cepstra[i] for i in sort_indices]    # sort the cepstra list
        final_labels = [final_labels[i] for i in sort_indices]      # sort the label list

    # SAVE Cepstra to files (features)
    FeatureExtractor.save_cepstra(final_cepstra, save_folder, exist_ok=True)

    # SAVE Transcripts to files (labels)
    pdtsc.save_labels([final_labels], save_folder, exist_ok=True)

    print('files transformed and saved into {}.'.format(os.path.abspath(save_folder)))


if __name__ == '__main__':
    # extracting audiofiles, transforming into cepstra and saving to separate folders
    feature_type = "MFSC"

    audio_folder = "../data/raw/audio/"
    transcript_folder = "../data/raw/transcripts/"
    save_folder = '../data/{}_deltas/'.format(feature_type)

    files = extract_filenames(audio_folder, transcript_folder)

    prepare_data(files, save_folder, feature_type="MFSC", energy=True, deltas=(2, 2), filter_nan=True)
