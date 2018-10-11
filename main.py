# import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from matplotlib import pyplot as plt


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


def make_frames(signal, frequency, width=0.025, stride=0.01):
    """divide the time-based signal into frames with specific width and stride
    :param signal: time-domain signal to be divided into frames
    :param width: the width of one frame in seconds
    :param stride: the stride at which the frames are made in seconds
    :return: array (frame_length,n_frames) of individual frames
    """

    sqn_len = len(signal)
    frame_len = int(width * frequency)
    stride_len = int(stride * frequency)
    n_frames = np.array(np.ceil((sqn_len - frame_len) / stride_len) + 1, dtype=np.int32)

    frames = [signal[i * stride_len:i * stride_len + frame_len].T for i in range(n_frames)]

    # last frame should be padded to same length as the other frames
    if len(frames[-1]) < frame_len:
        frames[-1] = np.pad(frames[-1], (0, frame_len - len(frames[-1])), 'constant', constant_values=0)
        print(np.shape(frames[-1]))
    else:
        last_frame = None

    return np.vstack(frames)


def hamming(frame):
    """Apply Hamming window on input frame"""
    return 0.54 - 0.46 * np.cos(2 * np.pi * frame / (len(frame) - 1))


def make_stereo(signal):
    """Makes a stereo signal (2 channels) from mono signal"""
    return np.vstack((signal, signal)).T


def short_time_ft(signal, frequency):
    """Apply short time fourier transform directly on the input signal"""
    return stft(signal, fs=frequency, window='hamming')


def fourier_transform(frames):
    """Apply fast fourier transform to frames of input signal"""
    n_fft = np.shape(frames)[1]
    return np.fft.fft(a=frames, n=n_fft, axis=1)[:, :round(n_fft/2)]


if __name__ == '__main__':
    frequency, signal = wavfile.read('data/saxophone-scale.wav')

    signal = extract_mono(signal)

    sig_len = np.shape(signal)[0]

    time = np.linspace(0, sig_len / frequency, sig_len)

    # applying pre-emphasis filter on signal
    pre_emphasised_signal = pre_emphasis(signal)

    # pre-emphasized signal to frames
    frames = make_frames(pre_emphasised_signal, frequency, width=0.005, stride=0.001)

    # apply Hamming window on every frame
#    frames = np.array([hamming(frame) for frame in frames])  # explicit solution
    frames *= np.hamming(np.shape(frames)[1])  # using numpy implementation of hamming window

    # apply FFT to the frames
    frames_fft = fourier_transform(frames)

    # the spectrogram is a square of the FFT
    frames_fft_squared = np.abs(frames_fft)**2

    print(np.shape(frames_fft))
    print(np.shape(frames_fft_squared))

    # TODO: Apply Short-Time Fourier-Transform on frames to transfer them to frequency domain
    t, f, STFT = short_time_ft(pre_emphasised_signal, frequency)
    # TODO: Filter Banks: spectrogram of the signal adjusted to fit human non-linear perception of sound (Mel-scale)

    plt.figure(1)
    ax1 = plt.subplot(411)
    plot_signal(time, signal, title='Soundwave signal from audiofile in time domain.')
    ax2 = plt.subplot(412)
    plot_signal(time, pre_emphasised_signal, title='Soundwave signal after applying pre-emphasis filter.')
    ax3 = plt.subplot(413)
    plt.pcolormesh(np.arange(np.shape(frames_fft)[0]),
                   np.arange(np.shape(frames_fft)[1]),
                   frames_fft_squared.T,
                   cmap='hot')
    ax4 = plt.subplot(414)
    plt.pcolormesh(f, t, np.abs(STFT), cmap='hot')
    plt.tight_layout()

    plt.show()

    stereo_pre_emph = make_stereo(pre_emphasised_signal)
    print(np.shape(stereo_pre_emph))  # TODO: remove print

    wavfile.write('data/preempth_saxophone.wav', frequency, stereo_pre_emph)
