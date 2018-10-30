
import numpy as np

from scipy.fftpack import dct

class MFCC:

    def __init__(self, data, fs, **kwargs):
        """

        :param data: list of 1d numpy arrays which represent the signal (audio)
        :param fs: sample rate of the signals in data
        :arg kwargs:
            :param framewidth: width of the frames that divide the data in seconds (default: 0.025)
            :param framestride: stride of the frames that divide the data in seconds (default: 0.01)
            :param fmin: starting frequency of the filterbanks (default: 300)
            :param fmax: ending frequency of the filterbanks (default: fs/2)
            :param nfft: length of the Fast Fourier Transformation (dafault: 512)
            :param nbanks: number of mel-scaled filterbanks applied on the STFTed frames (default: 26)
            :param cepstrums: slice of the final cepstra which are to be used as features (default: slice(1, 13))
        """
        self.data = data
        self.fs = fs                                                                     # Hz
        self.framewidth = kwargs['framewidth'] if 'framewidth' in kwargs else 0.025      # s
        self.framestride = kwargs['framestride'] if 'framestride' in kwargs else 0.01    # s
        self.fmin = kwargs['fmin'] if 'fmin' in kwargs else 300                          # Hz
        self.fmax = kwargs['fmax'] if 'fmax' in kwargs else fs/2                         # Hz
        self.nfft = kwargs['nfft'] if 'nfft' in kwargs else 512                          # 1
        self.nbanks = kwargs['nbanks'] if 'nbanks' in kwargs else 26                     # 1
        self.cepstrums = kwargs['cepstrums'] if 'cepstrums' in kwargs else slice(1, 13)  # 1

    def transform_data(self, deltas=(0, 0)):

        data = self.pre_emphasis(self.data)
        frames = self.make_frames(data, self.fs, self.framewidth, self.framestride)
        hamminged = self.hamming(frames)
        fft = self.fourier_transform(hamminged, self.nfft)
        power_fft = self.power_spectrum(fft)
        freq_idxs = self.mel_scaled_frequency_range()
        triangles = self.triangular_filterbanks(freq_idxs)
        log_sum = self.log_sum_of_filtered_frames(power_fft, triangles)
        mfcc = self.discrete_cosine_transform(log_sum)
        mfcc_standard = self.standardize(mfcc)

        if deltas[0]:
            d_mfcc = self.delta_multiple_inputs(mfcc, deltas[0])
            d_mfcc_standard = self.standardize(d_mfcc)
            d_mfcc_standard = self.pad_with_zeros(d_mfcc_standard, deltas[0])
        if deltas[1]:
            d2_mfcc = self.delta_multiple_inputs(self.delta_multiple_inputs(mfcc, deltas[0]), deltas[1])
            d2_mfcc_standard = self.standardize(d2_mfcc)
            d2_mfcc_standard = self.pad_with_zeros(d2_mfcc_standard, sum(deltas))

        if not any(deltas):
            return mfcc_standard
        elif all(deltas):
            return [np.hstack((mfcc_standard[i], d_mfcc_standard[i], d2_mfcc_standard[i])) for i in range(len(mfcc))]
        elif deltas[0]:
            return [np.hstack((mfcc_standard[i], d_mfcc_standard[i])) for i in range(len(mfcc))]
        elif deltas[1]:
            return [np.hstack((mfcc_standard[i], d2_mfcc_standard[i])) for i in range(len(mfcc))]



    def show_settings(self):
        attr_dict = vars(self)
        output = "fs = {} Hz\n" \
                 "framewidth = {} s\n" \
                 "framestride = {} s\n" \
                 "fmin = {} Hz\n" \
                 "fmax = {} Hz\n" \
                 "nfft = {} \n" \
                 "cepstrums = {} \n".format(*list(attr_dict.values())[1:])
        return output

    def pre_emphasis(self, data, alpha=0.95):
        """Applies preemphasis filter on the list of signals which boosts the high frequencies"""
        nr = len(data)
        pre_data = [np.array(0, dtype=np.float32)]*nr

        for i, row in enumerate(self.data):
            pre_data[i] = np.append(row[0], row[1:] - alpha * row[:-1])

        return pre_data

    @staticmethod
    def make_frames(data, fs, width, stride):
        """divide the time-based signal into frames with specific width and stride
        :param data: list of 1D numpy arrays of signals to be processed
        :param fs: sample rate of the signal
        :param width: the width of one frame in seconds
        :param stride: the stride at which the frames are made in seconds
        :return: list of numpy arrays with overall shape (nr,frame_length,n_frames)
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
    def hamming(framed_data):
        """Apply Hamming window on individual frames in rows of framed_data"""
        nr = len(framed_data)
        hamminged_data = [np.array(0, dtype=np.float32)] * nr

        for i, row in enumerate(framed_data):
            hamminged_data[i] = row*np.hamming(np.shape(row)[1])

        return hamminged_data

    @staticmethod
    def fourier_transform(framed_data, nfft=512):
        """Apply fast fourier transform to frames of a real input signal from rows of framed_data"""
        nr = len(framed_data)
        fft_transformed = [np.array(0, dtype=np.float32)] * nr

        for i, row in enumerate(framed_data):
            fft_transformed[i] = np.fft.rfft(a=row, n=nfft, axis=1)[:, :int(nfft/2 + 1)]

        return fft_transformed

    def power_spectrum(self, framed_fft):
        return [1 / self.nfft * (np.abs(row) ** 2) for row in framed_fft]

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

    @staticmethod
    def triangular_filterbanks(f_idxs):
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

        return filterbanks

    @staticmethod
    def log_sum_of_filtered_frames(framed_data, filters):
        """log10 of the sum of Dot product of the frames with filters in individual rows of framed_data"""
        return [np.log10(np.matmul(frames, filters.T)) for frames in framed_data]

    def discrete_cosine_transform(self, filtered_data):
        return [dct(row, type=2, axis=1, norm='ortho')[:, self.cepstrums] for row in filtered_data]

    @staticmethod
    def delta(c, order=2):
        """calculate the Delta of the 2D cepstral array c column-wise from order surrounding frames"""
        nrows, ncols = np.shape(c)
        tspan = range(order, nrows - order)

        dinp_arr = np.zeros((nrows - 2 * order, ncols))

        for i in range(ncols):
            dinp_arr[:, i] = np.array([sum(n * (c[t + n, i] - c[t - n, i]) for n in range(1, order + 1))
                                       / (2 * sum(n ** 2 for n in range(1, order + 1))) for t in tspan])

        return dinp_arr

    def delta_multiple_inputs(self, cepstral_data, order=2):
        return [self.delta(row, order) for row in cepstral_data]

    def pad_with_zeros(self, cepstral_data, pad_width=2):
        """pad individual arrays in cepstral_data list with pad_width zeros at both sides of first axis"""
        return [np.pad(inp_arr, [(pad_width, pad_width), (0, 0)], mode='constant') for inp_arr in cepstral_data]

    def standardize(self, cepstral_data):
        return [np.subtract(inp_arr, np.mean(inp_arr, axis=0)) / np.std(inp_arr, axis=0) for inp_arr in cepstral_data]

if __name__ == '__main__':
    data = [np.random.randn(np.random.randint(3000, 5000)) for _ in range(1000)]
    fs = 16000
    m = MFCC(data, fs)

    cepstra = m.transform_data()
    cepstra_d = m.transform_data(deltas=(2, 0))
    cepstra_2d = m.transform_data(deltas=(2, 2))
    print(np.shape(cepstra[0]))
    print(np.shape(cepstra_d[0]))
    print(np.shape(cepstra_2d[0]))
