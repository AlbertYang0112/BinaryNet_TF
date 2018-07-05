import unittest
import numpy as np
from main import *

class TestBinaryOp(unittest.TestCase):

    def setUp(self):
        self.sess = tf.Session()
        self.testFloatInput = tf.constant([[0.1, -0.4],[-0.2, 0.8]], dtype=tf.float32)
        self.testIntInput = tf.constant([[-1, 5], [0, 7]], dtype=tf.int32) 
        self.testBinaryInput = tf.constant([[-1, 1], [1, 1]], dtype=tf.int32)


    def tearDown(self):
        self.sess.close()

    
    def test_Binarize(self):
        binarized = Binarize(self.testFloatInput)
        output = self.sess.run(binarized)
        self.assertTrue((output == [[1, -1], [-1, 1]]).all())

    
    def test_ap2(self):
        approximate = ap2(self.testFloatInput)
        output = self.sess.run(approximate)
        self.assertTrue((output == [[2**-3, -2**-1],[-2**-2, 2**0]]).all())
    
    def test_ShiftBasedNormalization(self):
        acc = tf.nn.batch_normalization(self.testFloatInput, 
            mean=tf.constant([-0.05, 0.2]), 
            variance=tf.constant([0.0225, 0.36]), 
            offset=[0, 0], scale=[2, 2], 
            variance_epsilon=1e-5)
        shiftBN = ShiftBasedBatchNormalization(self.testFloatInput, 
            offset = 0, 
            scale = 2)
        outputAcc, outputShiftBN = self.sess.run([acc, shiftBN])
        print('Acc', outputAcc)
        print('ShiftBasedBN', outputShiftBN)
        meanAcc = np.mean(outputAcc)
        meanSBN = np.mean(outputShiftBN)
        self.assertAlmostEqual(meanAcc, 0)
        self.assertAlmostEqual(meanSBN, 0)
        self.assertAlmostEqual(np.mean(outputAcc - outputShiftBN), 0)

if __name__ == '__main__':
    unittest.main()