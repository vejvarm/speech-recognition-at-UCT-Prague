import time

from models.AcousticModel import AcousticModel

# TODO: feeding data in at inference -> determine what should be ran at __init__ when do_train is false
# https://stackoverflow.com/questions/50986886/how-to-inference-with-a-single-example-in-tensorflow-with-dataset-pipeline
# TODO: Bias gradients in LSTM are absurdly high (thousands)
# suffle the order of batches (keeps similar lengths in batch but adds variability through epochs)
# decaying learning rate
# TODO: Gradient masking on feed forward layers https://stackoverflow.com/questions/43364985/how-to-stop-gradient-for-some-entry-of-a-tensor-in-tensorflow
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
