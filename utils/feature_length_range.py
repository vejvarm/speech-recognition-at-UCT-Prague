import os
import shutil

from FeatureExtraction import FeatureExtractor
from DataLoader import DataLoader


def feature_length_range(load_dir, save_dir, min_frame_length=100, max_frame_length=3000, mode='copy',
                         feature_names='cepstrum', label_names='transcript'):
    """ Check individual files (features and their labels) in load_dir and copy/move those which satisfy the condition:
    min_frame_length <= feature_frame_len <= max_frame_length

    :param load_dir: folder from which to load features and their labels
    :param save_dir: folder to which copy/move the files which satisfy the condition above
    :param min_frame_length: lower bound of the feature frame length condition
    :param max_frame_length: upper bound of the feature frame length condition
    :param mode: 'copy'/'move' - condition satisfying files are copied/moved from load_dir to save_dir
    :param feature_names: sequence of symbols that can be used as common identifier for feature files
    :param label_names: sequence of symbols that can be used as common identifier for label files
    :return: None
    """

    # normalize the save directory path
    save_path = os.path.normpath(save_dir)

    folder_structure_gen = os.walk(load_dir)  # ('path_to_current_folder', [subfolders], ['files', ...])

    for folder in folder_structure_gen:
        path, subfolders, files = folder
        feat_file_names = [f for f in files if feature_names in f]
        label_file_names = [f for f in files if label_names in f]

        num_feats = len(feat_file_names)
        num_labels = len(label_file_names)

        assert num_feats == num_labels, 'There is {} feature files and {} label files (must be same).'.format(num_feats, num_labels)

        rel_path = os.path.relpath(path, load_dir)  # relative position of current subdirectory in regards to load_dir
        save_full_path = os.path.join(save_path, rel_path)  # folder/subfolder to which save files in save_dir

        # make subdirectories in save_dir
        os.makedirs(save_full_path, exist_ok=True)

        for i in range(num_feats):
            feat_load_path = os.path.join(path, feat_file_names[i])
            label_load_path = os.path.join(path, label_file_names[i])
            feat_save_path = os.path.join(save_full_path, feat_file_names[i])
            label_save_path = os.path.join(save_full_path, label_file_names[i])

            feat, _ = FeatureExtractor.load_cepstra(feat_load_path)
            feat_frame_len = feat[0][0].shape[0]

            if min_frame_length <= feat_frame_len <= max_frame_length:
                if mode == 'copy':
                    shutil.copy2(feat_load_path, feat_save_path)
                    print("Copied {} to {}".format(feat_load_path, feat_save_path))
                    shutil.copy2(label_load_path, label_save_path)
                    print("Copied {} to {}".format(label_load_path, label_save_path))
                elif mode == 'move':
                    os.rename(feat_load_path, feat_save_path)
                    print("Moved {} to {}".format(feat_load_path, feat_save_path))
                    os.rename(label_load_path, label_save_path)
                    print("Moved {} to {}".format(label_load_path, label_save_path))
                else:
                    raise ValueError("argument mode must be eiher 'copy' or 'move'")

        print("Finished.")


if __name__ == '__main__':
    min_frame_length = 100
    max_frame_length = 3000
    load_dir = "b:/!temp/PDTSC_MFSC_unigram_40_banks/"
    save_dir = "b:/!temp/PDTSC_MFSC_unigram_40_banks_min_{}_max_{}".format(min_frame_length, max_frame_length)

    feature_length_range(load_dir, save_dir, min_frame_length, max_frame_length, mode='move')
