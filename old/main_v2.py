import os
import numpy as np
# noinspection PyPackageRequirements

from matplotlib import pyplot as plt

from MFCC import MFCC
from DataLoader import PDTSCLoader


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


def samples_for_svk(sample=0, folder='./private/images/', dpi=300):
    pdtsc = PDTSCLoader(['data/pdtsc_003.ogg'], ['data/pdtsc_003.wdata'])
    labels = pdtsc.transcripts_to_labels()
    audio, fs = pdtsc.load_audio()

    audio = audio[0]
    fs = fs[0]

    mfcc = MFCC([audio[sample]], fs)
    cepstra = mfcc.transform_data()
    power_sfft = mfcc.power_sfft[0]
    log_sum = mfcc.log_sum[0]

    timespan = np.arange(len(audio[sample]))/fs
    framespan = np.arange(np.shape(power_sfft)[0]) * mfcc.framestride
    freqspan = np.arange(np.shape(power_sfft)[1]) / mfcc.nfft * fs

    # plot audio signal
    plt.figure()
    plot_signal(timespan, audio[sample], title='Audio signal sample no. {}'.format(sample))
    plt.savefig(folder + 'audiosignal.png', dpi=dpi)  # save the figure

    # plot power spectral density of audio signal
    plt.figure()
    plot_spectrum(framespan, freqspan, power_sfft, title='Periodogram of sample no. {}'.format(sample))
    plt.savefig(folder + 'power_spectrum.png', dpi=dpi)  # save the figure

    # plot mel scaled filterbanks
    plt.figure()
    plot_filters(freqspan, mfcc.filterbanks, title='Mel-scaled frequency filter bank'.format(sample))
    plt.savefig(folder + 'filterbank.png', dpi=dpi)  # save the figure

    # plot log10 of mel scaled filterbanks
    plt.figure()
    plot_logsum(framespan, mfcc.nbanks, log_sum, title='log10 of matmul(P,F) for sample no. {}'.format(sample))
    plt.savefig(folder + 'logsum.png', dpi=dpi)  # save the figure

    # plot final mfcc
    mfcc.plot_cepstra(cepstra, nplots=1)
    plt.savefig(folder + 'mfcc.png', dpi=dpi)  # save the figure

    print('transcript: ' + pdtsc.tokens[0][sample])
    print('labels: {}'.format(labels[0][sample]))

    plt.show()


if __name__ == '__main__':
    samples_for_svk(sample=3)
