import tensorflow as tf
binarizeOp = tf.load_op_library('lib/libbinarizeOp.so')
testInput = tf.random_normal([3, 3], mean = 0, stddev = 1)
with tf.Session() as sess:
    testInputPrinted = tf.Print(testInput, [testInput], 
                                message='Input Tensor',
                                summarize=9)
    binarized = binarizeOp.binarize(testInputPrinted)
    output = sess.run(binarized)
    print(output)