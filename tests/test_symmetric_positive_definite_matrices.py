"""Unit tests for sphere module."""

import geomstats.symmetric_positive_definite_matrices as spd_matrices

import numpy as np
import unittest


class TestSPDMatricesMethods(unittest.TestCase):
    def test_is_symmetric(self):
        sym_mat = np.array([[1, 2],
                            [2, 1]])
        self.assertTrue(spd_matrices.is_symmetric(sym_mat))

        not_a_sym_mat = np.array([[1., 0.6, -3.],
                                  [6., -7., 0.],
                                  [0., 7., 8.]])
        self.assertFalse(spd_matrices.is_symmetric(not_a_sym_mat))

    def matrix_to_vector_and_vector_to_matrix(self):
        sym_mat_1 = np.array([[1., 0.6, -3.],
                              [0.6, 7., 0.],
                              [-3., 0., 8.]])
        vector_1 = spd_matrices.matrix_to_vector(sym_mat_1)
        result_1 = spd_matrices.vector_to_matrix(vector_1)
        expected_1 = sym_mat_1

        self.assertTrue(np.allclose(result_1, expected_1))

    def test_group_log_and_exp(self):
        point_1 = 5 * np.eye(4)
        group_log_1 = spd_matrices.group_log(point_1)
        result_1 = spd_matrices.group_exp(group_log_1)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

    def test_riemannian_log_and_exp(self):
        ref_point_1 = np.array([[5., 0., 0.],
                                [0., 7., 2.],
                                [0., 2., 8.]])
        point_1 = np.array([[9., 0., 0.],
                            [0., 5., 0.],
                            [0., 0., 1.]])

        riem_log_1 = spd_matrices.riemannian_log(ref_point_1, point_1)
        result_1 = spd_matrices.riemannian_exp(ref_point_1, riem_log_1)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))


if __name__ == '__main__':
        unittest.main()
