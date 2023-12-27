import unittest
import numpy as np
from data_augmentations import *


class TestPacketNum2Nbytes(unittest.TestCase):

    def setUp(self):

        self.packet_num2nbytes = PacketNum2Nbytes()

    def test_transform(self):
        # Set up test data
        time_stamps = [0.1, 0.2, 0.3, 0.4, 0.5]
        sizes = [100, 200, 150, 180, 120]
        flowpic = np.asanyarray({'flowpic':{'time_stamps': time_stamps, 'sizes': sizes}}
                                {'info':{'image_dims':[32,32]}})
        result = self.packet_num2nbytes(flowpic)
        expected_result = np.array([[100, 200, 150, 180, 120]])

        # Perform the transform

        # Check if the result matches the expected result
        np.testing.assert_array_equal(result, expected_result)

    def test_name(self):
        self.assertEqual(self.packet_num2nbytes.name, 'PacketNum2Nbytes')


if __name__ == '__main__':
    unittest.main()
