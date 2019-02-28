from models.AcousticModel import AcousticModel

# TODO: fix exploding loss values at second epoch --> gradient_clipping
if __name__ == '__main__':
    config_path = "./config"
    ac_model = AcousticModel(config_path)

    ac_model.train(save_every=1)
