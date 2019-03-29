import tensorflow as tf
import numpy as np

from FeatureExtraction import FeatureExtractor
load_cepstra = FeatureExtractor.load_cepstra

if __name__ == '__main__':
    load_dir = '../data'
    cepstra = np.array(load_cepstra(load_dir))
    cepstra_tf = tf.convert_to_tensor()

    # create a tf.Dataset object from cepstra
    # tf.data.Dataset.from_tensors(cepstra_tf)