from models.AcousticModel import AcousticModel

# TODO: placeholder for grad_clip_val (should increase with time length?)
# TODO: suffle the order of batches (keeps similar lengths in batch but adds variability through epochs)
# TODO: Gradient masking on feed forward layers https://stackoverflow.com/questions/43364985/how-to-stop-gradient-for-some-entry-of-a-tensor-in-tensorflow
# TODO: https://www.tensorflow.org/api_docs/python/tf/custom_gradient
if __name__ == '__main__':
    config_path = "./config"
    ac_model = AcousticModel(config_path)

    ac_model.train(save_every=1)
