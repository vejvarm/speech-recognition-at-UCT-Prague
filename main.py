import time

from models.AcousticModel import AcousticModel

# TODO: efficient dataset pipeline (work with only paths to files until needed)
# https://www.tensorflow.org/tutorials/load_data/images

# __ DS2: https://arxiv.org/pdf/1512.02595.pdf __
# GRU or even simple RNN instead of LSTM
# 1D or 2D convolution as the first layer (with stride > 1 for reducing sequence length for RNN)
# TODO: Implement SortaGrad
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
    config_path = "d:/!private/Lord/Git/speech_recognition/data/results/03_convolutions/MFSC/rnn/cer[0.37]_2019-0422-232515_conv[64-128-256](bn_dp[0.1])_rnn[512-512]/config.json"
    audiofile = "d:/!private/Lord/Git/speech_recognition/data/results/best_so_far/"
    num_repeats = 1

    for _ in range(num_repeats):
        ac_model = AcousticModel(config_path)

        # ac_model.train(save_every=1)
        # t = time.time()
        ac_model.infer(audiofile)
        # print(time.time() - t)

