import os


def sort_by_feature_file_size(folder, feature_pattern='cepstrum', label_pattern='transcript'):
    """

    :param folder:
    :param feature_pattern:
    :param label_pattern:
    :return:
    """

    path_gen = os.walk(folder)

    file_size_list = []

    for path, subfolders, files in path_gen:
        fullpaths = []
        for file in files:
            if feature_pattern in file:
                label_fullpath = os.path.join(path, file.replace(feature_pattern, label_pattern))  # corresponding label path
                if os.path.exists(label_fullpath):
                    fullpaths.append((os.path.join(path, file), label_fullpath))
                else:
                    message = 'corresponding label file not found at path {}'.format(label_fullpath)
                    raise FileNotFoundError(message)

        file_size_list.extend(zip(fullpaths, [os.path.getsize(fp[0]) for fp in fullpaths]))

    sorted_file_size_list = sorted(file_size_list, key=lambda x: x[1])

    return sorted_file_size_list


def move_to_shard_folders(sorted_list, min_shard_size=1024, save_folder=None,
                          save_feature_pattern='cepstrum', save_label_pattern='transcript'):
    """

    :param sorted_list:
    :param min_shard_size: minimum size of folder shards in which the data is split in MBytes
    :param save_folder:
    :param save_feature_pattern:
    :param save_label_pattern:
    :return:
    """
    data_len = len(sorted_list)
    data_size = sum(sfs[1] for sfs in sorted_list)
    byte_min_shard_size = min_shard_size*1e6  # convert to Byte size
    max_num_shards = int(data_size//byte_min_shard_size + 1)
    num_data_digits = len(str(data_len))
    num_shard_digits = len(str(max_num_shards))

    current_shard_size = 0
    file_idx = 0
    shard_idx = 0

    if not save_folder:
        save_folder = os.path.dirname(sorted_list[0][0][0])
        print('Save folder implicitly set to {}'.format(save_folder))

    for (feature_path, label_path), size in sorted_list:
        current_shard_folder = os.path.join(save_folder, 'shard_{0:0{1}d}'.format(shard_idx, num_shard_digits))
        os.makedirs(current_shard_folder, exist_ok=True)
        new_feature_name = '{0}-{1:0{2}d}.npy'.format(save_feature_pattern, file_idx, num_data_digits)
        new_label_name = '{0}-{1:0{2}d}.npy'.format(save_label_pattern, file_idx, num_data_digits)
        os.rename(feature_path, os.path.join(current_shard_folder, new_feature_name))
        os.rename(label_path, os.path.join(current_shard_folder, new_label_name))
        if current_shard_size < byte_min_shard_size:
            file_idx += 1
            current_shard_size += size
        else:
            print('Files for shard number {} saved to folder {}'.format(shard_idx, current_shard_folder))
            file_idx = 0
            shard_idx += 1
            current_shard_size = 0


if __name__ == '__main__':
    folders = ['b:/!temp/ORAL_MFSC_unigram_40_banks_min_100_max_3000/train/',
               'b:/!temp/ORAL_MFSC_unigram_40_banks_min_100_max_3000/test/']
    shard_sizes = [2**10, 2**7]
    for i in range(len(folders)):
        sorted_file_list = sort_by_feature_file_size(folders[i])
        move_to_shard_folders(sorted_file_list, min_shard_size=shard_sizes[i])
