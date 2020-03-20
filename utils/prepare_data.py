import os

from itertools import compress

import numpy as np

from pysndfx import AudioEffectsChain
from FeatureExtraction import FeatureExtractor
from DataLoader import DataLoader, PDTSCLoader, OralLoader
from helpers import console_logger

LOGGER = console_logger(__name__, "DEBUG")


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


def prepare_data(files, save_folder, dataset="pdtsc", label_max_duration=10.0, speeds=(0.9, 1.0, 1.1),
                 feature_type="MFSC", bigrams=False, repeated=False, energy=True, deltas=(0, 0),
                 nbanks=40, filter_nan=True, sort=True):
    cepstra_length_list = []

    file_names = get_file_names(files)

    for speed in speeds:
        LOGGER.info(f"Create audio_transormer for speed {speed}")
        audio_transformer = (AudioEffectsChain().speed(speed))
        save_path = os.path.join(save_folder, f"{speed}/")
        LOGGER.debug(f"Current save_path: {save_path}")
        for i, file in enumerate(files):
            if dataset == "pdtsc":
                pdtsc = PDTSCLoader([file[0]], [file[1]], bigrams, repeated)
                labels = pdtsc.transcripts_to_labels()  # list of lists of 1D numpy arrays
                labels = labels[0]  # flatten label list
                audio_list, fs = pdtsc.load_audio()
                audio = audio_list[0]
                fs = fs[0]
                LOGGER.debug(f"Loaded PDTSC with fs {fs} from:\n \t audio_path: {file[0]}\n \t transcript_path: {file[1]}")
            elif dataset == "oral":
                oral = OralLoader([file[0]], [file[1]], bigrams, repeated)
                label_dict = oral.transcripts_to_labels(label_max_duration)  # Dict['file_name':Tuple[sents_list, starts_list, ends_list]]
                audio_dict, fs_dict = oral.load_audio()  # Dicts['file_name']

                labels = label_dict[file_names[i]]
                audio = audio_dict[file_names[i]]
                fs = fs_dict[file_names[i]]
                LOGGER.debug(f"Loaded ORAL with fs {fs} from:\n \t audio_path: {file[0]}\n \t transcript_path: {file[1]}")
            else:
                raise ValueError("'dataset' argument must be either 'pdtsc' or 'oral'")

            full_save_path = os.path.join(save_path, file_names[i])

            LOGGER.info(f"\tApplying SoX transormation on audio from {full_save_path}")
            for ii in range(len(audio)):
                LOGGER.debug(f"\t\t input.shape: {audio[ii].shape}")
                audio[ii] = audio_transformer(audio[ii])
                LOGGER.debug(f"\t\t output.shape: {audio[ii].shape}")

            LOGGER.info(f"\tApplying FeatureExtractor on audio")
            feat_ext = FeatureExtractor(audio, fs, feature_type=feature_type, energy=energy, deltas=deltas, nbanks=nbanks)
            cepstra = feat_ext.transform_data()  # list of 2D arrays

            # filter out cepstra which are containing nan values
            if filter_nan:
                LOGGER.info(f"\tFiltering out NaN values")
                # boolean list where False marks cepstra in which there is at least one nan value present
                mask_nan = [not np.isnan(cepstrum).any() for cepstrum in cepstra]

                # mask out cepstra and their corresponding labels with nan values
                cepstra = list(compress(cepstra, mask_nan))
                labels = list(compress(labels, mask_nan))

            # SAVE Cepstra to files (features)
            LOGGER.info(f"\tSaving cepstra to files")
            FeatureExtractor.save_cepstra(cepstra, full_save_path, exist_ok=True)
            LOGGER.debug(f"\t\tfull_save_path: {full_save_path}")

            # SAVE Transcripts to files (labels)
            LOGGER.info(f"\tSaving transcripts to files")
            if dataset == 'pdtsc':
                pdtsc.save_labels([labels], save_path, exist_ok=True)
            elif dataset == 'oral':
                label_dict[file_names[i]] = labels
                oral.save_labels(label_dict, save_path, exist_ok=True)
            else:
                raise ValueError("'dataset' argument must be either 'pdtsc' or 'oral'")

            LOGGER.info(f"\tChecking SAVE/LOAD consistency")
            loaded_cepstra, loaded_cepstra_paths = FeatureExtractor.load_cepstra(full_save_path)
            loaded_labels, loaded_label_paths = DataLoader.load_labels(full_save_path)

            # flatten the lists
            loaded_cepstra, loaded_cepstra_paths, loaded_labels, loaded_label_paths = (loaded_cepstra[0],
                                                                                       loaded_cepstra_paths[0],
                                                                                       loaded_labels[0],
                                                                                       loaded_label_paths[0])

            for j in range(len(cepstra)):
                if np.any(np.not_equal(cepstra[j], loaded_cepstra[j])):
                    raise UserWarning("Saved and loaded cepstra are not value consistent.")
                if dataset == 'pdtsc':
                    if np.any(np.not_equal(labels[j], loaded_labels[j])):
                        raise UserWarning("Saved and loaded labels are not value consistent.")
                elif dataset == 'oral':
                    if np.any(np.not_equal(labels[j][0], loaded_labels[j])):
                        raise UserWarning("Saved and loaded labels are not value consistent.")

                # add (cepstrum_path, label_path, cepstrum_length) tuple into collective list for sorting
                cepstra_length_list.append((loaded_cepstra_paths[j], loaded_label_paths[j], loaded_cepstra[j].shape[0]))

            LOGGER.debug(f'files from {file_names[i]} transformed and saved into {os.path.abspath(save_path)}.')

        # sort cepstra and labels by time length (number of frames)
        if sort:
            LOGGER.info(f"Sorting cepstra and labels by time length (number of frames)")
            sort_indices = np.argsort([c[2] for c in cepstra_length_list])  # indices which sort the lists by cepstra length
            cepstra_length_list = [cepstra_length_list[i] for i in sort_indices]  # sort the cepstra list

            num_digits = len(str(len(cepstra_length_list)))

            for idx, file in enumerate(cepstra_length_list):
                cepstrum_path, label_path, _ = file
                os.rename(cepstrum_path, "{0}/cepstrum-{1:0{2}d}.npy".format(save_path, idx, num_digits))
                os.rename(label_path, "{0}/transcript-{1:0{2}d}.npy".format(save_path, idx, num_digits))
            subfolders = next(os.walk(save_path))[1]
            for folder in subfolders:
                try:
                    os.rmdir(os.path.join(save_path, folder))
                except OSError:
                    LOGGER.warning("Folder {} is not empty! Can't delete.".format(os.path.join(save_path, folder)))


if __name__ == '__main__':
    # extracting audiofiles, transforming into cepstra and saving to separate folders
    dataset = "pdtsc"
    feature_type = "MFSC"
    label_type = "unigram"
    bigrams = True if label_type == "bigram" else False
    repeated = False
    energy = True
    deltas = (2, 2)
    nbanks = 40
    filter_nan = True
    sort = False

    audio_folder = "D:/Audio/Speech_Datasets/PDTSC/audio/"
    transcript_folder = "D:/Audio/Speech_Datasets/PDTSC/transcripts/"
    save_folder = 'B:/!temp/{}_{}_{}_{}_banks/'.format(dataset.upper(), feature_type, label_type, nbanks)

    files = get_file_paths(audio_folder, transcript_folder)

    prepare_data(files, save_folder, dataset=dataset, feature_type=feature_type,
                 bigrams=bigrams, repeated=repeated, energy=energy,
                 deltas=deltas, nbanks=nbanks, filter_nan=filter_nan, sort=sort)
