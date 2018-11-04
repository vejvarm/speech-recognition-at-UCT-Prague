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

#    mfcc.plot_cepstra(cepstra, nplots=1)

    # TODO: SAVE Cepstra to files (features)
    mfcc.save_cepstra(cepstra, './data/pdtsc_142', exist_ok=True)

    cepstra2 = mfcc.load_cepstra('./data/pdtsc_142')

    assert all([np.array_equal(c1, c2)
                for c1, c2 in zip(cepstra, cepstra2)]), 'Loaded data are not consistent with the saved data'

if __name__ == '__main__':
    main()
