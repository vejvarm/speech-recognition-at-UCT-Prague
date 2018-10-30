from unittest import TestCase

import numpy as np

from MFCC import MFCC


class TestMFCC(TestCase):
    def setUp(self):
        data = np.zeros((1000, 2000))
        self.mfcc = MFCC(data, 16000)


class TestInit(TestMFCC):

    def test_kwargs(self):
        self.assertEqual(self.mfcc.framewidth, 0.025)
        self.assertEqual(self.mfcc.framestride, 0.01)
        self.assertEqual(self.mfcc.fmin, 300)
        self.assertEqual(self.mfcc.fmax, self.mfcc.fs/2)
        self.assertEqual(self.mfcc.nfft, 512)
        self.assertEqual(self.mfcc.cepstrums, slice(1, 13))


class TestPreEmphasis(TestMFCC):

    def test_shape(self):
        pre_data = self.mfcc.pre_emphasis()
        self.assertEqual(self.mfcc.data.shape, pre_data.shape)


class TestHamming(TestMFCC):

    def test_shape(self):
        data = self.mfcc.data
        fs = self.mfcc.fs
        width = self.mfcc.framestride
        stride = self.mfcc.framewidth

        frames = self.mfcc.make_frames(data, fs, width, stride)
        hamminged_frames = self.mfcc.hamming(frames)

        self.assertEqual(np.shape(frames), np.shape(hamminged_frames))

    def test_values(self):
        data = self.mfcc.data
        fs = self.mfcc.fs
        width = self.mfcc.framestride
        stride = self.mfcc.framewidth

        frames = self.mfcc.make_frames(data, fs, width, stride)
        tested_hamming = self.mfcc.hamming(frames)[0]

        true_hamming = frames[0] * np.hamming(np.shape(frames[0])[1])

        self.assertTrue((tested_hamming == true_hamming).all())
