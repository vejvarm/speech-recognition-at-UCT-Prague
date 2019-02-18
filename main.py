from models.AcousticModel import AcousticModel

from matplotlib import pyplot as plt
import tensorflow as tf

if __name__ == '__main__':
    config_path = "./config"
    ac_model = AcousticModel(config_path)

    epochs = 10

    with tf.Session() as sess:
        # training dataset
        sess.run(tf.global_variables_initializer())
        # training dataset
        for epoch in range(epochs):
            print("EPOCH {}".format(epoch))
            total_train_loss = 0
            count_train = 0
            sess.run(ac_model.inputs["init_train"])
            print("_____TRAINING DATA_____")
            try:
                while True:
                    sess.run(ac_model.outputs["optimizer"])
                    ctc_loss, avg_loss, output = sess.run([ac_model.outputs["ctc_loss"],
                                                           ac_model.outputs["avg_loss"],
                                                           ac_model.outputs["ctc_output"]])
                    total_train_loss += avg_loss
                    count_train += 1
                    if count_train % 2 == 0:
                        print("BATCH {} | Avg. Loss {}".format(count_train, avg_loss))
            except tf.errors.OutOfRangeError:
                print("CTC Loss: {}".format(ctc_loss))

            total_test_loss = 0
            count_test = 0
            sess.run(ac_model.inputs["init_test"])
            print("_____TESTING DATA_____")
            try:
                while True:
                    ctc_loss, avg_loss, output = sess.run([ac_model.outputs["ctc_loss"],
                                                           ac_model.outputs["avg_loss"],
                                                           ac_model.outputs["ctc_output"]])
                    total_test_loss += avg_loss
                    count_test += 1
                    print("BATCH {} | Avg. Loss {}".format(count_test, avg_loss))
            except tf.errors.OutOfRangeError:
                print("CTC Loss: {}".format(ctc_loss))
