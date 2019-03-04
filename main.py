from models.AcousticModel import AcousticModel

# TODO: sequence-length adaptive learning rate, reduce learning rate as the sequences become longer
# TODO: Bias gradients in LSTM are absurdly high (thousands)
# TODO: suffle the order of batches (keeps similar lengths in batch but adds variability through epochs)
# TODO: decaying learning rate
# TODO: Gradient masking on feed forward layers https://stackoverflow.com/questions/43364985/how-to-stop-gradient-for-some-entry-of-a-tensor-in-tensorflow
# TODO: https://www.tensorflow.org/api_docs/python/tf/custom_gradient
if __name__ == '__main__':
    config_path = "./config.json"
    ac_model = AcousticModel(config_path)

    ac_model.train(save_every=1)
