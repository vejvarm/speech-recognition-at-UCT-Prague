import numpy as np

from matplotlib import pyplot as plt
from scipy.io import wavfile  # for loading WAV audio file format

from MFCC import MFCC

def plot_signal(time, signal, title=''):
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, 'major')


def plot_spectrum(frmspan, fspan, spectrum, title=''):
    # normalize spectrum to (0,1)
    spectrum_normal = spectrum / spectrum.max(axis=0)

    plt.pcolormesh(frmspan, fspan, spectrum_normal.T, cmap='inferno')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')


def plot_filters(fspan, filterbanks, title=''):
    plt.plot(fspan, filterbanks.T)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (1)')


def plot_logsum(framespan, nbanks, logsum, title=''):
    bankspan = np.arange(nbanks)
    plt.title(title)
    plt.pcolormesh(framespan, bankspan, logsum.T)
    plt.xlabel('Time (s)')
    plt.ylabel('Bank (1)')


if __name__ == '__main__':
    folder = '../private/images/mfcc/'
    dpi = 300
    sample_rate, audio = wavfile.read('../private/audio/auto.wav')
    audio = audio[int(0.3*sample_rate):int(0.7*sample_rate)]

    data = [audio]
    fs = sample_rate

    mfcc = MFCC(data, fs)

    cepstra = mfcc.transform_data()
    cepstraDelta = mfcc.transform_data(deltas=(2, 0))
    cepstraDelta2 = mfcc.transform_data(deltas=(2, 2))
    power_sfft = mfcc.power_sfft[0]
    log_sum = mfcc.log_sum[0]

    timespan = np.arange(len(audio)) / fs
    framespan = np.arange(np.shape(power_sfft)[0]) * mfcc.framestride
    freqspan = np.arange(np.shape(power_sfft)[1]) / mfcc.nfft * fs

    # plot audio signal
    plt.figure()
    plot_signal(timespan, audio, title='Raw audio signal')
    plt.savefig(folder + 'audiosignal.png', dpi=dpi)  # save the figure

    # plot power spectral density of audio signal
    plt.figure()
    plot_spectrum(framespan, freqspan, power_sfft, title='Periodogram of audio signal.')
    plt.savefig(folder + 'power_spectrum.png', dpi=dpi)  # save the figure

    # plot mel scaled filterbanks
    plt.figure()
    plot_filters(freqspan, mfcc.filterbanks, title='Mel-scaled frequency filter banks')
    plt.savefig(folder + 'filterbank.png', dpi=dpi, transparent=True)  # save the figure

    # plot log10 of mel scaled filterbanks
    plt.figure()
    plot_logsum(framespan, mfcc.nbanks, log_sum, title='log10 of matmul(P,F).')
    plt.savefig(folder + 'logsum.png', dpi=dpi)  # save the figure

    # plot final mfcc
    mfcc.plot_cepstra(cepstra, nplots=1)
    plt.savefig(folder + 'mfcc.png', dpi=dpi)  # save the figure

    # plot mfcc with deltas
    mfcc.plot_cepstra(cepstraDelta, nplots=1)
    plt.savefig(folder + 'mfccDelta.png', dpi=dpi)  # save the figure

    # plot mfcc with delta-deltas
    mfcc.plot_cepstra(cepstraDelta2, nplots=1)
    plt.savefig(folder + 'mfccDelta2.png', dpi=dpi)  # save the figure


    plt.show()
