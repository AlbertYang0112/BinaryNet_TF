import tensorflow as tf


def ShiftBasedMul(x, y):
    return tf.multiply(x, y)


def Binarize(input_tensor):
    return tf.sign(input_tensor)


def ap2(input_tensor):
    sign = tf.sign(input_tensor)
    r = tf.round(tf.log(tf.abs(input_tensor)) / tf.log(2.))
    approximate = tf.multiply(sign, tf.pow(2., r))
    return approximate


def ShiftBasedBatchNormalization(input_tensor, offset, scale, eps=1e-5):
    offset, scale, eps = map(float, (offset, scale, eps))
    avg = tf.reduce_mean(input_tensor, axis = 0)
    # avg = tf.Print(avg, [avg], message='SBN Input avg')
    centeredInput = input_tensor - avg
    approxCenteredInput = ap2(centeredInput)
    variance = tf.reduce_mean(
        ShiftBasedMul(centeredInput, approxCenteredInput),
        axis = 0
        )
    # variance = tf.Print(variance, [variance], message='SBN Input Var')
    div = tf.rsqrt(variance + eps)
    normalized = ShiftBasedMul(centeredInput, div)
    scaleAndShifted = ShiftBasedMul(ap2(scale), normalized) + offset
    return scaleAndShifted


def ShiftBasedAdaMax(prev_param, grad, prev_moment, perv_velocity,
    learning_rate, alpha, beta1, beta2):
    # Update biased 1st and 2nd moment estimates
    moment = tf.multiply(beta1, prev_moment) + \
        tf.multiply((1 - beta1), grad)
    velocity = tf.maximum(
        tf.multiply(beta2, perv_velocity), 
        tf.abs(grad))

    # Update parameters
    param = prev_param - tf.multiply(
        ShiftBasedMul(alpha, 1 - beta1), 
        ShiftBasedMul(moment, tf.pow(velocity, -1))
        )

    return param, moment, velocity
