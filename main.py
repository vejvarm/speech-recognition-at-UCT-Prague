# import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt

def plot_signal(time, signal, title=''):
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, 'major')

def pre_emphasis(signal, alpha=0.95):
    """Applies preemphasis filter on the signal which boosts the high frequencies"""
    return np.append(signal[0], signal[1:] - alpha*signal[:-1])

def extract_mono(signal):
    """Extract single channel from a stereo signal waveform"""
    return signal[:, 0]

def make_frames(signal, frequency, width=0.025, stride=0.01):
    """divide the signal into frames with specific width and stride

    :param signal: time-domain signal to be divided into frames
    :param width: the width of one frame in seconds
    :param stride: the stride at which the frames are made in seconds
    :return: array of individual frames
    """

    frame_length, frame_step = frequency*width, int(frequency*stride)
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    num_frames = int(np.ceil(
        float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)  # Pad Signal to make sure that all frames have equal number of samples
    #  without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    return pad_signal[indices.astype(np.int32, copy=False)]

def hamming(frame):
    """Apply Hamming window on input frame"""

    return 0.54 - 0.46*np.cos(2*np.pi*frame/(len(frame)-1))

def make_stereo(signal):
    """Makes a stereo signal (2 channels) from mono signal"""
    return np.vstack((signal, signal)).T


if __name__ == '__main__':
    frequency, signal = wavfile.read('data/ucisedobre.wav')

    signal = extract_mono(signal)

    sig_len = np.shape(signal)[0]

    time = np.linspace(0, sig_len/frequency, sig_len)

    # applying pre-emphasis filter on signal
    pre_emphasised_signal = pre_emphasis(signal)

    # pre-emhasized signal to frames
    frames = make_frames(pre_emphasised_signal, frequency, width=0.025, stride=0.01)

    print(np.shape(frames))  # TODO: remove print

    # apply Hamming window on every frame
    frames_hamming = np.array([hamming(frame) for frame in frames])  # explicit solution
    frames *= np.hamming(np.shape(frames)[1])  # using numpy implementation of hamming window

    print(np.shape(frames_hamming), np.shape(frames))  # TODO: remove print

    # TODO: Apply Short-Time Fourier-Transform on frames to transfer them to frequency domain
    # TODO: Filter Banks: spectrogram of the signal adjusted to fit human non-linear perception of sound (Mel-scale)

    plt.figure(1)
    ax1 = plt.subplot(211)
    plot_signal(time, signal, title='Soundwave signal from audiofile in time domain.')
    ax2 = plt.subplot(212)
    plot_signal(time, pre_emphasised_signal, title='Soundwave signal after applying pre-emphasis filter.')
    plt.tight_layout()

    plt.show()

    stereo_pre_emph = make_stereo(pre_emphasised_signal)
    print(np.shape(stereo_pre_emph))  # TODO: remove print

 #   wavfile.write('data/preempth_ucisedobre.wav', frequency, stereo_pre_emph)
