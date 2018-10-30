import numpy as np

from bs4 import BeautifulSoup

class DataLoader:

    def __init__(self, audiofiles, transcripts):
        self.audiofiles = audiofiles
        self.transcripts = transcripts
        self.audio = [np.array(0, dtype='float')]
        self.tokens = []  # array of strings which will be filled with tokens (words) from transcripts
        self.labels = []  # numeric array which will contain numeric representations of characters
        self.c2n_map = {'a':  1, 'á':  2, 'b':  3,  'c':  4, 'č':  5, 'd':  6, 'ď':  7, 'e':  8, 'é':  9, 'ě': 10,
                        'f': 11, 'g': 12, 'h': 13, 'ch': 14, 'i': 15, 'í': 16, 'j': 17, 'k': 18, 'l': 19, 'm': 20,
                        'n': 21, 'ň': 22, 'o': 23,  'ó': 24, 'p': 25, 'q': 26, 'r': 27, 'ř': 28, 's': 29, 'š': 30,
                        't': 31, 'ť': 32, 'u': 33,  'ú': 34, 'ů': 35, 'v': 36, 'w': 37, 'x': 38, 'y': 39, 'ý': 40,
                        'z': 41, 'ž': 42, ' ': 43}

    def chararr2num(self, chararr):
        """ Transform character arrays to list with numeric representations of the characters depending on their
        position in the czech alphabet.
        """
        return [self.c2n_map[c] for c in ' '.join(chararr).lower()]

class PDTSCLoader(DataLoader):

    def __init__(self, audiofiles, transcripts):
        super().__init__(audiofiles, transcripts)

    def load_transcripts(self):
        for file in self.transcripts:
            with open(file, 'r', encoding='utf8') as f:
                raw = f.read()

            soup = BeautifulSoup(raw, 'xml')

            lm_tags = soup.find_all(lambda tag: tag.name == 'LM' and tag.has_attr('id'))
            start_time_tags = [LM.find('start_time') for LM in lm_tags]
            end_time_tags = [LM.find('end_time') for LM in lm_tags]
            token_tags = [LM.find_all('token') for LM in lm_tags]

            # TODO: odstranění speciálních znaků (a čísel) z tokens
            self.tokens = [' '.join([token.text.lower() for token in tokens]) for tokens in token_tags]

    def load_audio(self):
        pass



if __name__ == '__main__':
    pdtsc = PDTSCLoader(1, ['pdtsc_142.wdata'])

    print(pdtsc.transcripts)
#    print(pdtsc.chararr2num(['Ahoj já jsem Martin', 'To je super', 'Já taky']))
    pdtsc.load_transcripts()
    print(pdtsc.tokens)
    print(pdtsc.chararr2num(pdtsc.tokens))


