import tensorflow as tf
import numpy as np

def target_list_to_sparse(target_list):
    """Convert list of target 1D numpy arrays to sparse tensor indices, values and shape"""
    indices = []
    values = []
    shape = [len(target_list), max(len(target) for target in target_list)]
    for batch_idx, target in enumerate(target_list):
        for frame_idx, frame in enumerate(target):
            indices.append([batch_idx, frame_idx])
            values.append(frame)
    print(indices[0:1])
    print(values[0:1])
    print(shape)
    return np.array(indices), np.array(values), np.array(shape)


max_time = 100
out_time = 30
batch_size = 10
num_classes = 10
sequence_length = [max_time]*batch_size

# inp = np.random.randint(0, num_classes-1, (max_time, batch_size, num_classes))
# inp = np.random.randn(max_time, batch_size, num_classes)
one = np.ones((max_time, batch_size, 1))
zer = np.zeros_like(one)
inp = np.dstack((zer, zer, one, zer, zer, zer, zer, zer, zer, zer))
print(len(inp[0, :, 1]))
inputs = tf.constant(inp, dtype=tf.float32)
# tar_list = [np.random.randint(0, num_classes-1, out_time)]*batch_size
tar_list = [2*np.ones((out_time,))]*batch_size

print(len(tar_list), np.shape(tar_list[0]))

indices, values, shape = target_list_to_sparse(tar_list)

indices = tf.constant(indices, dtype=tf.int64)
values = tf.constant(values, dtype=tf.int32)
dense_shape = tf.constant(shape, dtype=tf.int64)

labels = tf.sparse.SparseTensor(indices, values, dense_shape)

decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(inputs,
                                                           sequence_length,
                                                           beam_width=10,
                                                           top_paths=1,
                                                           merge_repeated=True)

decoded_dense = [tf.sparse.to_dense(path) for path in decoded]
dense_from_labels = tf.sparse.to_dense(labels)

loss = tf.nn.ctc_loss(labels=labels, inputs=inputs, sequence_length=sequence_length, time_major=True)
cost = tf.reduce_mean(loss)

with tf.Session() as sess:
    out_dense = sess.run(decoded_dense)  # (batch_size,time)
    print(out_dense)
    _cost = sess.run(cost)
    print(_cost)
#    out = sess.run(decoded)
#    print(out[0].indices)
#    print(out[0].values)

#    log_probs = sess.run(log_probabilities)
#    print(log_probs)

#    np.save('decoded_output.npy', out[0].values)
#    print(sess.run(dense_from_labels))
#    _labels = sess.run(labels)
#    print(_labels[0])

