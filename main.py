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

def make_stereo(signal):
    """Makes a stereo signal (2 channels) from mono signal"""
    return np.transpose(np.vstack((signal, signal)))


if __name__ == '__main__':
    sample_rate, signal = wavfile.read('data/ucisedobre.wav')

    signal = extract_mono(signal)

    sig_len = np.shape(signal)[0]

    time = np.linspace(0, sig_len/sample_rate, sig_len)

    # applying pre-emphasis filter on signal
    pre_emphasised_signal = pre_emphasis(signal)

    plt.figure(1)
    ax1 = plt.subplot(211)
    plot_signal(time, signal, title='Soundwave signal from audiofile in time domain.')
    ax2 = plt.subplot(212)
    plot_signal(time, pre_emphasised_signal, title='Soundwave signal after applying pre-emphasis filter.')
    plt.tight_layout()

    plt.show()

    stereo_pre_emph = make_stereo(pre_emphasised_signal)

 #   wavfile.write('data/preempth_ucisedobre.wav', sample_rate, stereo_pre_emph)
