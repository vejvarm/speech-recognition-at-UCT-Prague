import time

from models.AcousticModel import AcousticModel

# __ DS2: https://arxiv.org/pdf/1512.02595.pdf __
# GRU or even simple RNN instead of LSTM
# 1D or 2D convolution as the first layer (with stride > 1 for reducing sequence length for RNN)
# TODO: RNN batch normalisation
# TODO: connecting labels to bigrams (increasing vocab size but reducing transcript length allowing bigger cnn stride)

# __ FEATURES __
# MFSC or spectrogram instead of MFCC

# __ PARALLELISATION __
# TODO: Model parallelism (TOWERS: same model on multiple GPU's with different batches going through them)
# Two cards (same specs) --> double the batch size

# TODO: feeding data in at inference -> determine what should be ran at __init__ when do_train is false
# https://stackoverflow.com/questions/50986886/how-to-inference-with-a-single-example-in-tensorflow-with-dataset-pipeline
# Bias gradients in LSTM are absurdly high (thousands)
# suffle the order of batches (keeps similar lengths in batch but adds variability through epochs)
# decaying learning rate
# TODO: https://www.tensorflow.org/api_docs/python/tf/custom_gradient
# TODO: cyclic learning rate
if __name__ == '__main__':
    config_path = "./config.json"
    audiofile = "./data/dobry_den.wav"
    ac_model = AcousticModel(config_path)

    ac_model.train(save_every=1)
    # t = time.time()
    # ac_model.infer(audiofile)
    # print(time.time() - t)
