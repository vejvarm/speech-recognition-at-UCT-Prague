from os.path import isdir
from shutil import rmtree
from unittest import TestCase

import numpy as np

from FeatureExtraction import FeatureExtractor


class TestFeatureExtraction(TestCase):
    def setUp(self):
        data = [np.random.randn(np.random.randint(3000, 5000)) for _ in range(100)]
        self.mfcc = FeatureExtractor(data, 16000, feature_type="MFCC", energy=True, deltas=(2, 2))
        self.cepstra = self.mfcc.transform_data()
        self.data_temp_folder = './data/temp'


class TestInit(TestFeatureExtraction):

    def test_kwargs(self):
        self.assertEqual(self.mfcc.framewidth, 0.025)
        self.assertEqual(self.mfcc.framestride, 0.01)
        self.assertEqual(self.mfcc.fmin, 300)
        self.assertEqual(self.mfcc.fmax, self.mfcc.fs/2)
        self.assertEqual(self.mfcc.nfft, 512)
        self.assertEqual(self.mfcc.cepstrums, slice(1, 13))


class TestPreEmphasis(TestFeatureExtraction):

    def test_shape(self):
        pre_emph_data = self.mfcc.pre_emphasis(self.mfcc.data)
        for i in range(len(self.mfcc.data)):
            self.assertEqual(self.mfcc.data[i].shape, pre_emph_data[i].shape)


class TestHamming(TestFeatureExtraction):

    def test_shape(self):
        data = self.mfcc.data
        fs = self.mfcc.fs
        width = self.mfcc.framestride
        stride = self.mfcc.framewidth

        frames = self.mfcc.make_frames(data, fs, width, stride)
        hamminged_frames = self.mfcc.hamming(frames)

        for i in range(len(frames)):
            self.assertEqual(frames[i].shape, hamminged_frames[i].shape)

    def test_values(self):
        data = self.mfcc.data
        fs = self.mfcc.fs
        width = self.mfcc.framestride
        stride = self.mfcc.framewidth

        frames = self.mfcc.make_frames(data, fs, width, stride)
        tested_hamming = self.mfcc.hamming(frames)[0]

        true_hamming = frames[0] * np.hamming(np.shape(frames[0])[1])

        self.assertTrue((tested_hamming == true_hamming).all())


class TestSaveLoad(TestFeatureExtraction):

    def test_save_load_consistency(self):

        if isdir(self.data_temp_folder):
            print("Warning: Temp directory was not empty. The data got overwritten.")
            rmtree(self.data_temp_folder, ignore_errors=True)

        self.mfcc.save_cepstra(cepstral_data=self.cepstra, folder=self.data_temp_folder, exist_ok=True)

        self.assertTrue(all([np.array_equal(c1, c2)
                            for c1, c2 in zip(self.cepstra, self.mfcc.load_cepstra(self.data_temp_folder))]))


