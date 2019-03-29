import os

import numpy as np

from scipy.fftpack import dct
from matplotlib import pyplot as plt


class FeatureExtractor:

    def __init__(self, data, fs, feature_type='MFSC', **kwargs):
        """

        :param data: list of 1d numpy arrays which represent the signal (audio)
        :param fs: sample rate of the signals in data
        :param feature_type: (str) type of the features ('MFSC' or 'MFCC')
        :arg kwargs:
            :param alpha: weighting coefficient of pre-emphasis filter (default: 0.95)
            :param framewidth: width of the frames that divide the data in seconds (default: 0.025)
            :param framestride: stride of the frames that divide the data in seconds (default: 0.01)
            :param energy: (bool) whether to include energy feature or not (default: True)
            :param fmin: starting frequency of the filterbanks (default: 300)
            :param fmax: ending frequency of the filterbanks (default: fs/2)
            :param nfft: length of the Fast Fourier Transformation (dafault: 512)
            :param nbanks: number of mel-scaled filterbanks applied on the STFTed frames (default: 26)
            :param cepstrums: slice of the final cepstra which are to be used as features (default: slice(1, 13))
            :param deltas: Tuple[int] containing span of deltas and delta-deltas
        """
        self.data = data
        self.fs = fs                                                                     # Hz
        self.type = feature_type
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.95                      # float
        self.framewidth = kwargs['framewidth'] if 'framewidth' in kwargs else 0.025      # float
        self.framestride = kwargs['framestride'] if 'framestride' in kwargs else 0.01    # float
        self.energy = kwargs['energy'] if 'energy' in kwargs else True                   # bool
        self.fmin = kwargs['fmin'] if 'fmin' in kwargs else 64                           # float (Hz)
        self.fmax = kwargs['fmax'] if 'fmax' in kwargs else fs/2                         # float (Hz)
        self.nfft = kwargs['nfft'] if 'nfft' in kwargs else 512                          # int
        self.nbanks = kwargs['nbanks'] if 'nbanks' in kwargs else 26                     # int
        self.cepstrums = kwargs['cepstrums'] if 'cepstrums' in kwargs else slice(1, 13)  # Slice
        self.deltas = kwargs['deltas'] if 'deltas' in kwargs else (0, 0)                 # Tuple[int]

        # initialize containers
        self.stft = (np.asarray(0, dtype=np.float32), )               # tuple for STFTed frames
        self.power_stft = (np.asarray(0, dtype=np.float32),)          # tuple for power density of STFTed frames
        self.filterbanks = np.zeros((self.nbanks, self.nfft//2 + 1))  # filters to be applied to power_stft
        self.log_sum = (np.asarray(0, dtype=np.float32),)             # log10 of matmul(power_stft, filterbanks)

    def transform_data(self):

        data = self.pre_emphasis(self.data, self.alpha)
        frames = self.make_frames(data, self.fs, self.framewidth, self.framestride)
        hamminged = self.hamming(frames)
        fft = self.fourier_transform(hamminged, self.nfft)
        power_stft = self.power_spectrum(fft)
        freq_idxs = self.mel_scaled_frequency_range()
        triangles = self.triangular_filterbanks(freq_idxs)
        features = self.log_sum_of_filtered_frames(power_stft, triangles)
        if self.type == 'MFCC':
            features = self.discrete_cosine_transform(features)
        if self.energy:
            energy = self.calculate_energy(frames)
            features = [np.hstack((features[i], energy[i])) for i in range(len(features))]
        features_standard = self.standardize(features)

        if self.deltas[0]:
            d_features = self.delta_multiple_inputs(features, self.deltas[0])
            d_features_standard = self.standardize(d_features)
        if self.deltas[1]:
            d2_features = self.delta_multiple_inputs(self.delta_multiple_inputs(features, self.deltas[0]), self.deltas[1])
            d2_features_standard = self.standardize(d2_features)

        if not any(self.deltas):
            return features_standard
        elif all(self.deltas):
            return [np.hstack((features_standard[i], d_features_standard[i], d2_features_standard[i]))
                    for i in range(len(features))]
        elif self.deltas[0]:
            return [np.hstack((features_standard[i], d_features_standard[i])) for i in range(len(features))]
        elif self.deltas[1]:
            return [np.hstack((features_standard[i], d2_features_standard[i])) for i in range(len(features))]

    def show_settings(self):
        attr_dict = vars(self)
        output = "fs = {} Hz\n" \
                 "type = {}\n" \
                 "framewidth = {} s\n" \
                 "framestride = {} s\n" \
                 "fmin = {} Hz\n" \
                 "fmax = {} Hz\n" \
                 "nfft = {} \n" \
                 "nbanks = {}\n" \
                 "cepstrums = {} \n".format(*list(attr_dict.values())[1:])
        return output

    @staticmethod
    def pre_emphasis(data, alpha=0.95):
        """Applies preemphasis filter on the list of signals which boosts the high frequencies"""
        nr = len(data)
        pre_data = [np.array(0, dtype=np.float32)]*nr

        for i, row in enumerate(data):
            pre_data[i] = np.append(row[0], row[1:] - alpha * row[:-1])

        return pre_data

    @staticmethod
    def make_frames(data, fs, width, stride):
        """divide the time-based signal into frames with specific width and stride
        :param data: list of 1D numpy arrays of signals to be processed
        :param fs: sample rate of the signal
        :param width: the width of one frame in seconds
        :param stride: the stride at which the frames are made in seconds
        :return: list of numpy arrays with overall shape (nr,n_frames,frame_length)
        """
        nr = len(data)
        framed_data = [np.array(0, dtype=np.float32)]*nr

        for i, row in enumerate(data):
            sgn_len = len(row)
            frame_len = int(width * fs)
            stride_len = int(stride * fs)
            n_frames = int(np.ceil((sgn_len - frame_len) / stride_len) + 1)

            frames = [row[i * stride_len:i * stride_len + frame_len].T for i in range(n_frames)]

            # last frame should be padded to same length as the other frames
            if len(frames[-1]) < frame_len:
                frames[-1] = np.pad(frames[-1], (0, frame_len - len(frames[-1])), 'constant', constant_values=0)

            framed_data[i] = np.vstack(frames)

        return framed_data

    @staticmethod
    def calculate_energy(framed_data):
        """calculate energy of the frames, which is a sum of squares over the length of the frame"""
        frames_energy = []
        for row in framed_data:
            frames_energy.append(np.sum(np.square(row), axis=1, dtype=np.float32).reshape((-1, 1)))

        return frames_energy

    @staticmethod
    def hamming(framed_data):
        """Apply Hamming window on individual frames in rows of framed_data"""
        nr = len(framed_data)
        hamminged_data = [np.array(0, dtype=np.float32)] * nr

        for i, row in enumerate(framed_data):
            hamminged_data[i] = row*np.hamming(row.shape[1])

        return hamminged_data

    def fourier_transform(self, framed_data, nfft=512):
        """Apply fast fourier transform to frames of a real input signal from rows of framed_data"""
        nr = len(framed_data)
        fft_transformed = [np.array(0, dtype=np.float32)] * nr

        for i, row in enumerate(framed_data):
            fft_transformed[i] = np.fft.rfft(a=row, n=nfft, axis=1)[:, :int(nfft/2 + 1)]

        self.stft = fft_transformed

        return fft_transformed

    def power_spectrum(self, framed_fft):
        self.power_stft = tuple(1 / self.nfft * (np.abs(row) ** 2) for row in framed_fft)
        return self.power_stft

    @staticmethod
    def mel_scale(freq_input):
        """Apply mel-scale formula on the input array of frequencies"""
        return 2595 * np.log10(1 + freq_input / 700)

    @staticmethod
    def inverse_mel_scale(mel_input):
        """Apply inverse operation to mel-scale formula to get frequencies from input array of mels"""
        return 700 * (10 ** (mel_input / 2595) - 1)

    def mel_scaled_frequency_range(self):
        """

        :return: indices of the mel-scaled frequencies
        """
        mmin = self.mel_scale(self.fmin)
        mmax = self.mel_scale(self.fmax)

        # linearly space n_banks values between mmin and mmax (to get n_banks+2 points)
        mels = np.linspace(mmin, mmax, self.nbanks+2)

        # calculate the inverse of mel-scale to get back the frequencies which will be the bounds of the triangular filters
        freqs = [self.inverse_mel_scale(mel) for mel in mels]

        # calculate the indices which are closest to the desired frequencies for the filters
        freqs_idxs = np.floor((self.nfft + 1) * np.divide(freqs, self.fs)).astype(np.int32)

        return freqs_idxs

    def triangular_filterbanks(self, f_idxs):
        """Calculate triangular filterbanks at desired indices.
        Each filterbank starts at f_idxs(i), linearly goes to 1 at f_idxs(i+1)
        and then goes linearly back to zero at f_idxs(i+2).
        """

        nbanks = len(f_idxs) - 2
        filterbanks = np.zeros((nbanks, f_idxs[-1] + 1))

        for i in range(nbanks):
            start = np.zeros(f_idxs[i])
            line_up = np.linspace(0, 1, f_idxs[i + 1] - f_idxs[i] + 1)
            line_down = np.linspace(1, 0, f_idxs[i + 2] - f_idxs[i + 1] + 1)
            end = np.zeros(f_idxs[-1] - f_idxs[i + 2])

            filterbanks[i, :] = np.hstack((start, line_up, line_down[1:], end))

        self.filterbanks = filterbanks

        return filterbanks

    def log_sum_of_filtered_frames(self, framed_data, filters):
        """log10 of the sum of Dot product of the frames with filters in individual rows of framed_data"""
        self.log_sum = tuple(np.log10(np.matmul(frames, filters.T)) for frames in framed_data)
        return self.log_sum

    def discrete_cosine_transform(self, filtered_data):
        return [dct(row, type=2, axis=1, norm='ortho')[:, self.cepstrums] for row in filtered_data]

    @staticmethod
    def delta(c, order=2):
        """calculate the Delta of the 2D cepstral array c column-wise from order surrounding frames

        :return (ndarray) dinp_arr [nrows, ncols] OR dinp_arr [nrows, ncols] of NaNs if nrows <= 2*order
        """
        nrows, ncols = np.shape(c)
        tspan = range(order, nrows - order)
        dnrows = tspan.stop - tspan.start  # number of rows (frames) in delta array

        if dnrows <= 0:
            dinp_arr = np.full((nrows, ncols), np.nan)  # return array of nans
            print("Deltas can't be calculated, returned array of NaNs for filtering purposes.")
            return dinp_arr
        else:
            dinp_arr = np.zeros((nrows, ncols))
            for i in range(ncols):
                dinp_arr[order:nrows - order, i] = np.array([sum(
                    n * (c[t + n, i] - c[t - n, i]) for n in range(1, order + 1)) / (
                                                                         2 * sum(n ** 2 for n in range(1, order + 1)))
                                                             for t in tspan])
            return dinp_arr

    def delta_multiple_inputs(self, cepstral_data, order=2):
        return [self.delta(cepstrum, order) for cepstrum in cepstral_data]

    @staticmethod
    def pad_with_zeros(cepstral_data, pad_width=2):
        """pad individual arrays in cepstral_data list with pad_width zeros at both sides of first axis"""
        return [np.pad(inp_arr, [(pad_width, pad_width), (0, 0)], mode='constant') for inp_arr in cepstral_data]

    @staticmethod
    def standardize(cepstral_data):
        return [np.subtract(inp_arr, np.mean(inp_arr, axis=0)) / np.std(inp_arr, axis=0) + 1e-8
                for inp_arr in cepstral_data]

    @staticmethod
    def plot_cepstra(cepstral_data, nplots=3, framestride=0.01):
        """plot first nplots  mfcc from cepstral_data into nplots sepparate figures"""
        assert isinstance(cepstral_data, list), "cepstral_data should be a list (of 2D MFCC numpy arrays)"
        assert isinstance(cepstral_data[0], np.ndarray), "cepstral_data list should contain 2D MFCC numpy arrays"
        for i in range(nplots):
            tspan = np.arange(np.shape(cepstral_data[i])[0]) * framestride
            ncepstra = np.arange(np.shape(cepstral_data[i])[1], dtype=np.int8)
            plt.figure()
            plt.pcolormesh(tspan, ncepstra, cepstral_data[i].T, cmap='rainbow')
            plt.title('Mel-frequency coefficients of sample no. {}'.format(i))
            plt.xlabel('Time (s)')
            plt.ylabel('coefficients (features)')


    @staticmethod
    def save_cepstra(cepstral_data, folder, exist_ok=False):
        """save mfcc from cepstral_data list to separate .npy files into specified folder"""
        assert isinstance(cepstral_data, list), "cepstral_data should be a list (of 2D MFCC numpy arrays)"
        assert isinstance(cepstral_data[0], np.ndarray), "cepstral_data list should contain 2D MFCC numpy arrays"
        try:
            os.makedirs(folder, exist_ok=exist_ok)
        except OSError:
            print('Folder already exists. Please select another folder or set exist_ok to True.')
            return
        ndigits = len(str(len(cepstral_data)))  # n zeroes to pad the name with in order to keep the correct order
        for i, array in enumerate(cepstral_data):
            np.save('{0}/cepstrum-{1:0{2}d}.npy'.format(folder, i, ndigits), array)

    @staticmethod
    def load_cepstra(folder):
        """load mfcc from cepstrum-###.npy files from specified folder (or subfolders if present)
        :param folder: string path leading to the folder with cepstra files

        :return list of lists of 2D numpy arrays, list of lists of strings with paths to files
        """
        # if the folder contains subfolders, load data from all subfolders
        cepstra = []
        path_list = []
        subfolders = [os.path.join(folder, subfolder) for subfolder in next(os.walk(folder))[1]]

        # if there are no subfolders in the provided folder, look for the transcripts directly in folder
        if not subfolders:
            subfolders.append(folder)

        for sub in subfolders:
            files = [os.path.splitext(f) for f in os.listdir(sub) if
                     os.path.isfile(os.path.join(sub, f))]
            paths = [os.path.abspath(os.path.join(sub, ''.join(file)))
                     for file in files if 'cepstrum' in file[0] and file[-1] == '.npy']  # load only .npy files
            subcepstra = [np.load(path) for path in paths]
            cepstra.append(subcepstra)
            path_list.append(paths)  # load only .npy files
        return cepstra, path_list


if __name__ == '__main__':
    data = [np.random.randn(np.random.randint(3000, 5000)) for _ in range(1000)]
    fs = 16000
    m = FeatureExtractor(data, fs, feature_type='MFSC', energy=True, nbanks=40, deltas=(2, 2))

    cepstra_2d = m.transform_data()
    m.plot_cepstra(cepstra_2d, nplots=3)
    plt.show()
