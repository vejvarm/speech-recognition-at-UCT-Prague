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

def main():
    pdtsc = PDTSCLoader(['data/pdtsc_142.ogg'], ['data/pdtsc_142.wdata'])
    labels = pdtsc.load_transcripts()
    audio, fs = pdtsc.load_audio()

    mfcc = MFCC(audio[0], fs)  # TODO: make MFCC work for more audiofiles
    cepstra = mfcc.transform_data()

    plt.figure(1)
    plt.pcolormesh(cepstra[5])  # TODO: def plot_cepstra
    plt.show()

    # TODO: SAVE Cepstra to files (features)

if __name__ == '__main__':
    main()
