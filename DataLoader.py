import numpy as np
import re
import soundfile as sf  # for loading OGG audio file format

from bs4 import BeautifulSoup

class DataLoader:

    def __init__(self, audiofiles, transcripts):
        self.audiofiles = audiofiles
        self.transcripts = transcripts
        self.audio = [np.array(0, dtype='float')]
        self.starts = [[]]*len(self.transcripts)  # list of lists of starting times of the sentences
        self.ends = [[]]*len(self.transcripts)    # list of lists of ending times of the sentences
        self.tokens = [[]]*len(self.transcripts)  # list of lists of tokens (sentences) from transcripts
        self.labels = [[]]*len(self.transcripts)  # list of arrays which will contain numeric representations of characters
        self.c2n_map = {'a':  0, 'á':  1, 'b':  2,  'c':  3, 'č':  4, 'd':  5, 'ď':  6, 'e':  7, 'é':  8, 'ě':  9,
                        'f': 10, 'g': 11, 'h': 12, 'ch': 13, 'i': 14, 'í': 15, 'j': 16, 'k': 17, 'l': 18, 'm': 19,
                        'n': 20, 'ň': 21, 'o': 22,  'ó': 23, 'p': 24, 'q': 25, 'r': 26, 'ř': 27, 's': 28, 'š': 29,
                        't': 30, 'ť': 31, 'u': 32,  'ú': 33, 'ů': 34, 'v': 35, 'w': 36, 'x': 37, 'y': 38, 'ý': 39,
                        'z': 40, 'ž': 41, ' ': 42}

    def char2num(self, sentlist):
        """ Transform list of sentences (tokens) to list of lists with numeric representations of the
        characters depending on their position in the czech alphabet.
        """
        return [[self.c2n_map[c] for c in chars.lower()] for chars in sentlist]

class PDTSCLoader(DataLoader):

    def __init__(self, audiofiles, transcripts):
        super().__init__(audiofiles, transcripts)

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

            # process the start times from start_time tags
            self.starts[i] = [start.text for start in start_time_tags if start]

            # process the ending times of sentences from end_time tags
            self.ends[i] = [end.text for end in end_time_tags if end]

            # process the tokens from token tags
            regexp = r'[^A-Za-záčďéěíňóřšťúůýž]+'  # find all non alphabetic characters
            tokens = [' '.join([re.sub(regexp, '', token.text.lower()) for token in tokens])
                      for tokens in token_tags]  # joining sentences and removing special and numeric chars
            self.tokens[i] = [token for token in tokens if token]  # removing empty strings

    def load_audio(self):
        signal, sample_rate = sf.read(self.audiofiles)
        pass



if __name__ == '__main__':
    pdtsc = PDTSCLoader(1, ['data/pdtsc_142.wdata'])

    print(pdtsc.transcripts)
#    print(pdtsc.char2num(['Ahoj já jsem Martin', 'To je super', 'Já taky']))
    pdtsc.load_transcripts()
    print(pdtsc.starts[0])
    print(pdtsc.tokens[0])
    print(pdtsc.char2num(pdtsc.tokens[0]))


