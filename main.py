import tensorflow as tf


def Binarize(input_tensor):
    return tf.sign(input_tensor)


def ap2(input_tensor):
    sign = tf.sign(input_tensor)
    ap = sign 
def ShiftBasedBatchNormalization(input_tensor):
    avg = tf.reduce_mean(input_tensor, axis = 1)
    centeredInput = input_tensor - avg
    std = 