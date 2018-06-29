import tensorflow as tf
zero_out_module = tf.load_op_library('lib/libtestop.so')
with tf.Session() as sess:
    print(zero_out_module.zero_out([[1,2],[9,8]]).eval())