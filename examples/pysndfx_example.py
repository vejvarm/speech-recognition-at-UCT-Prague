from pysndfx import AudioEffectsChain  # https://github.com/carlthome/python-audio-effects
from librosa import load

if __name__ == '__main__':
    fx = (AudioEffectsChain().speed(0.9))

    load()
