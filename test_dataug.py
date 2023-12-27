import unittest
import numpy as np
from data_augmentations import *


class TestPacketNum2Nbytes(unittest.TestCase):

    def setUp(self):

        self.packet_num2nbytes = PacketNum2Nbytes()

    def test_transform(self):
        # Set up test data
        time_stamps = [1000, 2000, 3000, 78000, 167000]
        sizes = [200, 210, 220, 500, 1300]
        flowpic = {
            'flowpic': {},
            'info': {
                'image_dims': [32, 32],
                'start_time': time_stamps[0],
                'time_stamps': time_stamps,
                'sizes': sizes
            }
        }
        
        expected_result = np.zeros((32, 32))
        expected_result[0][4] = 630
        expected_result[1][10] = 500
        expected_result[3][27] = 1300

        # Perform the transform
        result = self.packet_num2nbytes(flowpic)
        # print_hist(result)
        # Check if the result matches the expected result
        np.testing.assert_array_equal(result, expected_result)

    def test_name(self):
        self.assertEqual(self.packet_num2nbytes.name, 'PacketNum2Nbytes')


if __name__ == '__main__':
    unittest.main()
