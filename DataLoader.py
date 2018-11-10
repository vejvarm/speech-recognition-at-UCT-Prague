import re
import os

# from icu import LocaleData
import numpy as np
import soundfile as sf  # for loading OGG audio file format

from bs4 import BeautifulSoup


class DataLoader:

    def __init__(self, audiofiles, transcripts):
        """ Initialize DataLoader() object

        :param audiofiles: list of paths to audio files
        :param transcripts: list of paths to transcripts of the audio files
        """
        self.audiofiles = audiofiles
        self.transcripts = transcripts
        self.audio = [[np.array(0, dtype=np.float32)]]*len(self.audiofiles)
        self.fs = np.zeros((len(self.audiofiles)), dtype=np.uint16)            # sampling rates of the loaded audio files
        self.starts = [np.array(0, dtype=np.float32)]*len(self.transcripts)    # list of lists of starting times of the sentences
        self.ends = [np.array(0, dtype=np.float32)]*len(self.transcripts)      # list of lists of ending times of the sentences
        self.tokens = [[]]*len(self.transcripts)    # list of lists of tokens (sentences) from transcripts
        self.labels = [[np.array(0, dtype=np.uint8)]]*len(self.transcripts)  # list of arrays which will contain numeric representations of characters
        # self.c2n_map = {char: i for i, char in enumerate(string.alpha)}
        self.c2n_map = {'a':  0, 'á':  1, 'b':  2,  'c':  3, 'č':  4, 'd':  5, 'ď':  6, 'e':  7, 'é':  8, 'ě':  9,
                        'f': 10, 'g': 11, 'h': 12, 'ch': 13, 'i': 14, 'í': 15, 'j': 16, 'k': 17, 'l': 18, 'm': 19,
                        'n': 20, 'ň': 21, 'o': 22,  'ó': 23, 'p': 24, 'q': 25, 'r': 26, 'ř': 27, 's': 28, 'š': 29,
                        't': 30, 'ť': 31, 'u': 32,  'ú': 33, 'ů': 34, 'v': 35, 'w': 36, 'x': 37, 'y': 38, 'ý': 39,
                        'z': 40, 'ž': 41, ' ': 42}
        self.n2c_map = {val: idx for idx, val in self.c2n_map.items()}

    def char2num(self, sentlist):
        """ Transform list of sentences (tokens) to list of lists with numeric representations of the
        characters depending on their position in the czech alphabet.
        """
        arraylist = [np.asarray([self.c2n_map[c] for c in chars.lower()], dtype=np.uint8) for chars in sentlist]
        for i in range(len(arraylist)):
            ch_idcs = [(r.start(), r.end() - 1) for r in re.finditer('ch', sentlist[i])]

            # change arraylist at ch_idcs starts to number for symbol 'ch'
            mask_change = np.zeros(len(arraylist[i]), dtype=bool)
            mask_change[[tup[0] for tup in ch_idcs]] = True
            arraylist[i][mask_change] = self.c2n_map['ch']

            # remove elements after added numbers for symbol 'ch'
            mask_delete = np.ones(len(arraylist[i]), dtype=bool)
            mask_delete[[tup[1] for tup in ch_idcs]] = False
            arraylist[i] = arraylist[i][mask_delete]

        return arraylist

    def num2char(self, arraylist):
        """ Transform list of numpy arrays with chacater numbers to list of sentences """

        return [''.join([self.n2c_map[o] for o in arr]) for arr in arraylist]

    @staticmethod
    def extract_channel(signal, channel_number):
        """Extract single channel from a multi-channel (stereo) signal"""
        try:
            return signal[:, channel_number]
        except IndexError:
            return signal


class PDTSCLoader(DataLoader):

    def __init__(self, audiofiles, transcripts):
        super().__init__(audiofiles, transcripts)

    @staticmethod
    def time2secms(timelist):
        """Convert list of times in format hh:mm:ss.ms to numpy array of ss.ms format"""

        ssmsarray = np.zeros_like(timelist, dtype=np.float32)

        for i, time in enumerate(timelist):
            (hh, mm, ssms) = np.float32(time.split(':'))

            ssmsarray[i] = hh*3600 + mm*60 + ssms

        return ssmsarray

    def load_transcripts(self):
        for i, file in enumerate(self.transcripts):
            with open(file, 'r', encoding='utf8') as f:
                raw = f.read()

            soup = BeautifulSoup(raw, 'xml')

            # extract relevant tags from the soup
            lm_tags = soup.find_all(lambda tag: tag.name == 'LM' and tag.has_attr('id'))
            start_time_tags = [LM.find('start_time') for LM in lm_tags]
            end_time_tags = [LM.find('end_time') for LM in lm_tags]
            token_tags = [LM.find_all('token') for LM in lm_tags]

            # process the tokens from token tags
            regexp = r'[^A-Za-záéíóúýčďěňřšťůž{ch}]+'  # find all non alphabetic characters (Czech alphabet)
            tokens = [' '.join([re.sub(regexp, '', token.text.lower()) for token in tokens])
                      for tokens in token_tags]  # joining sentences and removing special and numeric chars

            empty_idcs = [i for i, token in enumerate(tokens) if not token]  # getting indices of empty tokens

            # removing empty_idcs from starts, ends and tokens
            start_time_tags = [tag for i, tag in enumerate(start_time_tags) if i not in empty_idcs]
            end_time_tags = [tag for i, tag in enumerate(end_time_tags) if i not in empty_idcs]
            tokens = [token for i, token in enumerate(tokens) if i not in empty_idcs]

            # save the start times, ent times and tokens to instance variables
            self.starts[i] = self.time2secms([start.text for start in start_time_tags])
            self.ends[i] = self.time2secms([end.text for end in end_time_tags])
            self.tokens[i] = tokens

            assert len(self.starts[i]) == len(self.ends[i]), "start times and end times don't have the same length"
            assert len(self.ends[i]) == len(self.tokens[i]), "there is different number of tokens than end times"

            # convert characters in tokens to numeric values representing their position in the czech alphabet
            self.labels[i] = self.char2num(self.tokens[i])

        return self.labels

    def save_labels(self, folder='./data/', exist_ok=False):
        """
        Save labels of transcripts to specified folder under folders with names equal to name of the transcrips files
        """
        if not self.labels[0]:
            print('The labels have not been generated yet. Please call load_transcripts class function first.')
            return

        # get names of the loaded transcript files and use them as subfolder names
        subfolders = tuple(os.path.splitext(os.path.basename(transcript))[0] for transcript in self.transcripts)

        try:
            for subfolder in subfolders:
                os.makedirs(os.path.join(folder, subfolder), exist_ok=exist_ok)
        except OSError:
            print('Subfolders already exist. Please set exist_ok to True if you want to save into them anyway.')
            return

        for idx in range(len(self.labels)):
            ndigits = len(str(len(self.labels[idx])))  # zeroes to pad the name with in order to keep the correct order
            fullpath = os.path.join(folder, subfolders[idx])
            for i, array in enumerate(self.labels[idx]):
                np.save('{0}/transcript-{1:0{2}d}.npy'.format(fullpath, i, ndigits), array)

    def load_audio(self):
        for i, file in enumerate(self.audiofiles):
            signal, self.fs[i] = sf.read(file)

            signal = self.extract_channel(signal, 0)  # convert signal from stereo to mono by extracting channel 0

            tstart = 0
            tend = signal.shape[0]/self.fs[i]
            tstep = 1/self.fs[i]
            tspan = np.arange(tstart, tend, tstep, dtype=np.float32)

            # find indices corresponding to the start and end times of the transcriptions
            starts_idcs = np.asarray(*[np.searchsorted(tspan, start) for start in self.starts], dtype=np.int32)
            ends_idcs = np.asarray(*[np.searchsorted(tspan, end) for end in self.ends], dtype=np.int32)

            # split the signal to intervals (starts_idcs[j], ends_idcs[j])
            # self.audio[i] = [signal[st_idx:ed_idx] for st_idx, ed_idx in zip(starts_idcs, ends_idcs)]
            self.audio[i] = [signal[starts_idcs[j]:ends_idcs[j]] for j in range(starts_idcs.shape[0])]

        return self.audio, self.fs

    @staticmethod
    def save_audio(file, audio, fs):
        sf.write(file, audio, fs)


if __name__ == '__main__':
#    pass
    pdtsc = PDTSCLoader(['data/pdtsc_142.ogg', 'data/pdtsc_001.ogg'], ['data/pdtsc_142.wdata', 'data/pdtsc_001.wdata'])
#    out = pdtsc.char2num(['chacha to je chalupa', 'achichouvej to je bolest', 'jako by se nechumelilo'])
#    print(out)
#    print(pdtsc.num2char(out))
#    print(pdtsc.transcripts)
#    print(pdtsc.char2num(['Ahoj já jsem Martin', 'To je super', 'Já taky']))
    pdtsc.load_transcripts()
#    print(pdtsc.labels)
#    pdtsc.load_audio()
#    print(pdtsc.starts[0][0])
#    print(pdtsc.ends[0][0])
#    print(pdtsc.tokens[0][0])
    print(pdtsc.labels[0][0])
#    print(pdtsc.audio[0][0])
#    pdtsc.save_audio('./data/test_saved.ogg', pdtsc.audio[0][1], pdtsc.fs[0])
    pdtsc.save_labels('./data', exist_ok=True)


