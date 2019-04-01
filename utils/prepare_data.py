import os

from itertools import compress

import numpy as np

from FeatureExtraction import FeatureExtractor
from DataLoader import PDTSCLoader


def get_file_paths(audio_folder, transcript_folder):
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


def get_file_names(files):
    return [os.path.splitext(os.path.split(file[0])[1])[0] for file in files]


def prepare_data(files, save_folder, feature_type="MFSC", energy=True, deltas=(0, 0), nbanks=40, filter_nan=True, sort=True):
    cepstra_length_list = []

    file_names = get_file_names(files)

    for i, file in enumerate(files):
        pdtsc = PDTSCLoader([file[0]], [file[1]])
        labels = pdtsc.transcripts_to_labels()  # list of lists of 1D numpy arrays
        labels = labels[0]  # flatten label list
        audio, fs = pdtsc.load_audio()

        full_save_path = os.path.join(save_folder, file_names[i])

        feat_ext = FeatureExtractor(audio[0], fs, feature_type=feature_type, energy=energy, deltas=deltas, nbanks=nbanks)
        cepstra = feat_ext.transform_data()  # list of 2D arrays

        # filter out cepstra which are containing nan values
        if filter_nan:
            # boolean list where False marks cepstra in which there is at least one nan value present
            mask_nan = [not np.isnan(cepstrum).any() for cepstrum in cepstra]

            # mask out cepstra and their corresponding labels with nan values
            cepstra = list(compress(cepstra, mask_nan))
            labels = list(compress(labels, mask_nan))

        # SAVE Cepstra to files (features)
        FeatureExtractor.save_cepstra(cepstra, full_save_path, exist_ok=True)

        # SAVE Transcripts to files (labels)
        pdtsc.save_labels([labels], save_folder, exist_ok=True)

        # __Checking SAVE/LOAD consistency__
        loaded_cepstra, loaded_cepstra_paths = FeatureExtractor.load_cepstra(full_save_path)
        loaded_labels, loaded_label_paths = pdtsc.load_labels(full_save_path)

        # flatten the lists
        loaded_cepstra, loaded_cepstra_paths, loaded_labels, loaded_label_paths = (loaded_cepstra[0],
                                                                                   loaded_cepstra_paths[0],
                                                                                   loaded_labels[0],
                                                                                   loaded_label_paths[0])

        for j in range(len(cepstra)):
            if np.any(np.not_equal(cepstra[j], loaded_cepstra[j])):
                raise UserWarning("Saved and loaded cepstra are not value consistent.")
            if np.any(np.not_equal(labels[j], loaded_labels[j])):
                raise UserWarning("Saved and loaded labels are not value consistent.")

            # add (cepstrum_path, label_path, cepstrum_length) tuple into collective list for sorting
            cepstra_length_list.append((loaded_cepstra_paths[j], loaded_label_paths[j], loaded_cepstra[j].shape[0]))

        print(cepstra_length_list)

        print('files from {} transformed and saved into {}.'.format(file_names[i], os.path.abspath(save_folder)))

    # sort cepstra and labels by time length (number of frames)
    if sort:
        sort_indices = np.argsort([c[2] for c in cepstra_length_list])  # indices which sort the lists by cepstra length
        cepstra_length_list = [cepstra_length_list[i] for i in sort_indices]  # sort the cepstra list

        print(cepstra_length_list)

        num_digits = len(str(len(cepstra_length_list)))

        for idx, file in enumerate(cepstra_length_list):
            cepstrum_path, label_path, _ = file
            os.rename(cepstrum_path, "{0}/cepstrum-{1:0{2}d}.npy".format(save_folder, idx, num_digits))
            os.rename(label_path, "{0}/transcript-{1:0{2}d}.npy".format(save_folder, idx, num_digits))
        subfolders = next(os.walk(save_folder))[1]
        for folder in subfolders:
            try:
                os.rmdir(os.path.join(save_folder, folder))
            except OSError:
                print("Folder {} is not empty! Can't delete.".format(os.path.join(save_folder, folder)))


if __name__ == '__main__':
    # extracting audiofiles, transforming into cepstra and saving to separate folders
    feature_type = "MFSC"

    audio_folder = "C:/!temp/raw_debug/audio/"
    transcript_folder = "C:/!temp/raw_debug/transcripts/"
    save_folder = 'C:/!temp/{}_debug/'.format(feature_type)

    files = get_file_paths(audio_folder, transcript_folder)

    prepare_data(files, save_folder, feature_type="MFSC", energy=True, deltas=(2, 2), nbanks=40, filter_nan=True)
