# import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from matplotlib import pyplot as plt
from scipy.fftpack import dct


def plot_signal(time, signal, title=''):
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, 'major')


def pre_emphasis(signal, alpha=0.95):
    """Applies preemphasis filter on the signal which boosts the high frequencies"""
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


def extract_mono(signal):
    """Extract single channel from a stereo signal waveform"""
    try:
        return signal[:, 0]
    except IndexError:
        return signal


def make_frames(signal, sample_rate, width=0.025, stride=0.01):
    """divide the time-based signal into frames with specific width and stride
    :param signal: time-domain signal to be divided into frames
    :param width: the width of one frame in seconds
    :param stride: the stride at which the frames are made in seconds
    :return: array (frame_length,n_frames) of individual frames
    """

    sqn_len = len(signal)
    frame_len = int(width * sample_rate)
    stride_len = int(stride * sample_rate)
    n_frames = np.array(np.ceil((sqn_len - frame_len) / stride_len) + 1, dtype=np.int32)

    frames = [signal[i * stride_len:i * stride_len + frame_len].T for i in range(n_frames)]

    # last frame should be padded to same length as the other frames
    if len(frames[-1]) < frame_len:
        frames[-1] = np.pad(frames[-1], (0, frame_len - len(frames[-1])), 'constant', constant_values=0)

    return np.vstack(frames)


def hamming(frame):
    """Apply Hamming window on input frame"""
    return 0.54 - 0.46 * np.cos(2 * np.pi * frame / (len(frame) - 1))


def make_stereo(signal):
    """Makes a stereo signal (2 channels) from mono signal"""
    return np.vstack((signal, signal)).T


def short_time_ft(signal, sample_rate, width=0.025, stride=0.01, n_fft=2048):
    """Apply short time fourier transform directly on the input signal"""
    frame_len = int(width * sample_rate)
    stride_len = int(stride * sample_rate)
    return stft(signal, fs=sample_rate, window='hamming', nfft=n_fft, nperseg=frame_len, noverlap=stride_len)


def fourier_transform(frames, n_fft=2048):
    """Apply fast fourier transform to frames of a real input signal"""
    return np.fft.rfft(a=frames, n=n_fft, axis=1)[:, :np.floor(n_fft/2 + 1).astype(np.int32)]

def mel_scale(freq_input):
    """Apply mel-scale formula on the input array of frequencies"""
    return 2595*np.log10(1 + freq_input/700)

def inverse_mel_scale(mel_input):
    """Apply inverse operation to mel-scale formula to get frequencies from input array of mels"""
    return 700*(10**(mel_input/2595) - 1)

def triangular_filterbanks(f_idxs):
    """Calculate triangular filterbanks at desired indices.
    Each filterbank starts at f_idxs(i), linearly goes to 1 at f_idxs(i+1)
    and then goes linearly back to zero at f_idxs(i+2).
    """

    n_banks = len(f_idxs)-2
    filterbanks = np.zeros((n_banks, f_idxs[-1]+1))

    for i in range(n_banks):

        start = np.zeros(f_idxs[i])
        line_up = np.linspace(0, 1, f_idxs[i+1] - f_idxs[i] + 1)
        line_down = np.linspace(1, 0, f_idxs[i+2] - f_idxs[i+1] + 1)
        end = np.zeros(f_idxs[-1] - f_idxs[i+2])

        filterbanks[i, :] = np.hstack((start, line_up, line_down[1:], end))

    return filterbanks


def log_sum_of_filtered_frames(frames, filters):
    """log10 of the sum of Dot product of the frames with filters"""
    return np.log10(np.matmul(frames, filters.T))

def discrete_cosine_transform(x):
    """DCT of x row-wise"""

    N = np.shape(x)[1]
    D = np.zeros_like(x)

    for i, row in enumerate(x):
        D[i, :] = [sum(row[n]*np.cos(np.pi/N*(n + 0.5)*k) for n in range(N)) for k in range(N)]

    return D


if __name__ == '__main__':
    sample_rate, signal = wavfile.read('data/saxophone-scale.wav')
    signal = signal[0:int(3.5 * sample_rate)]  # TODO: Don't forget that you only take the first 3.5 sec

    print('fs = {}'.format(sample_rate))  # TODO: Remove print

    signal = extract_mono(signal)

    sig_len = np.shape(signal)[0]

    time = np.linspace(0, sig_len / sample_rate, sig_len)

    # applying pre-emphasis filter on signal
    pre_emphasised_signal = pre_emphasis(signal)

    # pre-emphasized signal to frames
    frames = make_frames(pre_emphasised_signal, sample_rate, width=0.025, stride=0.01)

    # apply Hamming window on every frame
#    frames = np.array([hamming(frame) for frame in frames])  # explicit solution
    frames *= np.hamming(np.shape(frames)[1])  # using numpy implementation of hamming window

    print('frames: (L,R) = {}'.format(frames.shape))  # TODO: Remove print

    # apply FFT to the frames
    n_fft = 512
    frames_fft = fourier_transform(frames, n_fft)

    print('sfft: (R,Nfft/2+1) = {}'.format(frames_fft.shape))  # TODO: Remove print

    # the Periodogram estimate of the power spectrum
    power_spectrum = 1/n_fft*(np.abs(frames_fft)**2)

    # Compact approach - Apply Short-Time Fourier-Transform directly to signal to transfer them to frequency domain
    f, t, STFT = short_time_ft(pre_emphasised_signal, sample_rate)

    # TODO: Filter Banks: spectrogram of the signal adjusted to fit human non-linear perception of sound (Mel-scale)
    # calculate mel-scale maximum and minimum for desired frequency range
    f_min = 300  # Hz
    f_max = sample_rate/2  # Hz
    m_min = mel_scale(f_min)
    m_max = mel_scale(f_max)

    # TODO: Make this to function which returns freqs_idxs
    # linearly space 26 values between m_min and m_max (to get 28 points)
    n_banks = 28  # amount of filters to be used in the filter banks
    mels = np.linspace(m_min, m_max, n_banks)

    # calculate the inverse of mel-scale to get back the frequencies which will be the bounds of the triangular filters
    freqs = [inverse_mel_scale(mel) for mel in mels]

    # calculate the indices which are closest to the desired frequencies for the filters
    freqs_idxs = np.floor((n_fft+1)*np.divide(freqs, sample_rate)).astype(np.int32)

    # calculate filterbanks (triangular filters at freqs_idxs)
    filterbanks = triangular_filterbanks(freqs_idxs)

    print('fbanks: (B-2,Nfft/2+1) = {}'.format(filterbanks.shape))  # TODO: Remove print

    # apply individual filterbanks to each of the power spectrum frames
    log_energies = log_sum_of_filtered_frames(power_spectrum, filterbanks)

    print('logE: (R,B-2) = {}'.format(log_energies.shape))  # TODO: Remove print

    # apply discrete fourier transform (DCT) to log_energies
#    D = discrete_cosine_transform(log_energies)
    n_cepstr = 12
    D = dct(log_energies, type=2, axis=1, norm='ortho')[:, 1:n_cepstr+1]  # Keep 2-13

    print('DCT(logE): (R,C) = {}'.format(D.shape))  # TODO: Remove print

    # calculate frequency and time span for the axis
    tspan = np.arange(np.shape(power_spectrum)[0]) / sample_rate*n_fft
    fspan = np.arange(np.shape(power_spectrum)[1])/n_fft*sample_rate

    # plot the Mel-Scale frequency filterbanks
    plt.figure(1)
    plt.plot(fspan, filterbanks.T)
    plt.title('Mel-scale filterbanks')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (1)')

    # plot the signal, pre-emphasized signal and the spectrum from STFT
    plt.figure(2)
    ax1 = plt.subplot(411)
    plot_signal(time, signal, title='Soundwave signal from audiofile in time domain.')
    ax2 = plt.subplot(412)
    plot_signal(time, pre_emphasised_signal, title='Soundwave signal after applying pre-emphasis filter.')
    ax3 = plt.subplot(413)
    plt.pcolormesh(tspan,
                   fspan,
                   power_spectrum.T,
                   cmap='rainbow')
    ax4 = plt.subplot(414)
    plt.pcolormesh(t, f, 1/n_fft*(np.abs(STFT)**2), cmap='rainbow')
    plt.tight_layout()

    plt.figure(3)
    plt.pcolormesh(D.T, cmap='rainbow')

    plt.show()

    stereo_pre_emph = make_stereo(pre_emphasised_signal)

#    wavfile.write('data/preempth_saxophone.wav', sample_rate, stereo_pre_emph)
