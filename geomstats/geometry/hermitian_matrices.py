"""The vector space of Hermitian matrices.

Lead author: Yann Cabanes.
"""

import logging

import geomstats.backend as gs
import geomstats.vectorization
from geomstats import algebra_utils
from geomstats.geometry.base import ComplexVectorSpace
from geomstats.geometry.complex_matrices import ComplexMatrices, ComplexMatricesMetric


class HermitianMatrices(ComplexVectorSpace):
    """Class for the vector space of Hermitian matrices of size n.

    Parameters
    ----------
    n : int
        Integer representing the shapes of the matrices: n x n.
    """

    def __init__(self, n, **kwargs):
        kwargs.setdefault("metric", ComplexMatricesMetric(n, n))
        super(HermitianMatrices, self).__init__(
            shape=(n, n), default_point_type="matrix", **kwargs
        )
        self.n = n
        self.dim = n**2

    def _create_basis(self):
        """Compute the basis of the vector space of symmetric matrices."""
        basis = []
        for row in gs.arange(self.n):
            for col in gs.arange(row, self.n):
                if row == col:
                    indices = [(row, row)]
                    values = [1.0 + 0j]
                    basis.append(gs.array_from_sparse(indices, values, (self.n,) * 2))
                else:
                    indices = [(row, col), (col, row)]
                    values = [1.0 + 0j, 1.0 + 0j]
                    basis.append(gs.array_from_sparse(indices, values, (self.n,) * 2))
                    values = [1j, -1j]
                    basis.append(gs.array_from_sparse(indices, values, (self.n,) * 2))
        basis = gs.stack(basis)
        return basis

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a matrix is Hermitian.

        Parameters
        ----------
        point : array-like, shape=[.., n, n]
            Point to test.
        atol : float
            Tolerance to evaluate equality with the transpose.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        belongs = super(HermitianMatrices, self).belongs(point)
        if gs.any(belongs):
            is_hermitian = ComplexMatrices.is_hermitian(point, atol)
            return gs.logical_and(belongs, is_hermitian)
        return belongs

    @staticmethod
    def projection(point):
        """Make a matrix Hermitian, by averaging with its transconjugate.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        herm : array-like, shape=[..., n, n]
            Symmetric matrix.
        """
        return ComplexMatrices.to_hermitian(point)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample a Hermitian matrix using a uniform distribution in a box.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Side of hypercube support of the uniform distribution.
            Optional, default: 1.0

        Returns
        -------
        point : array-like, shape=[..., n, n]
           Sample.
        """
        cdtype = gs.get_default_cdtype()
        size = self.shape
        if n_samples != 1:
            size = (n_samples,) + self.shape
        point = gs.cast(
            bound * (gs.random.rand(*size) - 0.5) * 2**0.5,
            dtype=cdtype,
        ) + 1j * gs.cast(
            bound * (gs.random.rand(*size) - 0.5) * 2**0.5,
            dtype=cdtype,
        )
        return ComplexMatrices.to_hermitian(point)

    @staticmethod
    def to_vector(mat):
        """Convert a Hermitian matrix into a vector.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        vec : array-like, shape=[..., n(n+1)/2]
            Vector.
        """
        if not gs.all(ComplexMatrices.is_hermitian(mat)):
            logging.warning("non-Hermitian matrix encountered.")
        mat = ComplexMatrices.to_hermitian(mat)
        return gs.triu_to_vec(mat)

    @staticmethod
    @geomstats.vectorization.decorator(["vector", "else"])
    def from_vector(vec, dtype=None):
        """Convert a vector into a Hermitian matrix.

        Parameters
        ----------
        vec : array-like, shape=[..., n(n+1)/2]
            Vector.
        dtype : dtype, {gs.complex64, gs.complex128}
            Data type object to use for the output.
            Optional. Default: gs.complex128.

        Returns
        -------
        mat : array-like, shape=[..., n, n]
            Hermitian matrix.
        """
        if dtype is None:
            dtype = gs.get_default_cdtype()

        vec_dim = vec.shape[-1]
        mat_dim = (gs.sqrt(8.0 * vec_dim + 1) - 1) / 2
        if mat_dim != int(mat_dim):
            raise ValueError(
                "Invalid input dimension, it must be of the form"
                "(n_samples, n * (n + 1) / 2)"
            )
        mat_dim = int(mat_dim)
        shape = (mat_dim, mat_dim)
        mask = 2 * gs.ones(shape) - gs.eye(mat_dim)
        indices = list(zip(*gs.triu_indices(mat_dim)))
        vec = gs.cast(vec, dtype)
        upper_triangular = gs.stack(
            [gs.array_from_sparse(indices, data, shape) for data in vec]
        )
        mat = ComplexMatrices.to_hermitian(upper_triangular) * gs.cast(mask, dtype)
        return mat

    @classmethod
    def expm(cls, mat):
        """
        Compute the matrix exponential for a Hermitian matrix.

        Parameters
        ----------
        mat : array_like, shape=[..., n, n]
            Hermitian matrix.

        Returns
        -------
        exponential : array_like, shape=[..., n, n]
            Exponential of mat.
        """
        n = mat.shape[-1]
        dim_3_mat = gs.reshape(mat, [-1, n, n])
        expm = cls.apply_func_to_eigvals(dim_3_mat, gs.exp)
        expm = gs.reshape(expm, mat.shape)
        return expm

    @classmethod
    def powerm(cls, mat, power):
        """
        Compute the matrix power.

        Parameters
        ----------
        mat : array_like, shape=[..., n, n]
            Hermitian matrix with non-negative eigenvalues.
        power : float, list
            Power at which mat will be raised. If a list of powers is passed,
            a list of results will be returned.

        Returns
        -------
        powerm : array_like, shape=[..., n, n]
            Matrix power of mat.
        """
        if isinstance(power, list):
            power_ = [lambda ev, p=p: gs.power(ev, p) for p in power]
        else:

            def power_(ev):
                return gs.power(ev, power)

        return cls.apply_func_to_eigvals(mat, power_, check_positive=False)

    @staticmethod
    def apply_func_to_eigvals(mat, function, check_positive=False):
        """
        Apply function to eigenvalues and reconstruct the matrix.

        Parameters
        ----------
        mat : array_like, shape=[..., n, n]
            Hermitian matrix.
        function : callable, list of callables
            Function to apply to eigenvalues. If a list of functions is passed,
            a list of results will be returned.
        check_positive : bool
            Whether to check positivity of the eigenvalues.
            Optional. Default: False.

        Returns
        -------
        mat : array_like, shape=[..., n, n]
            Hermitian matrix.
        """
        eigvals, eigvecs = gs.linalg.eigh(mat)
        if check_positive and gs.any(gs.cast(eigvals, gs.get_default_dtype()) < 0.0):
            try:
                name = function.__name__
            except AttributeError:
                name = function[0].__name__

            logging.warning("Negative eigenvalue encountered in %s", name)

        return_list = True
        if not isinstance(function, list):
            function = [function]
            return_list = False
        reconstruction = []
        transconj_eigvecs = ComplexMatrices.transconjugate(eigvecs)
        for fun in function:
            eigvals_f = fun(eigvals)
            eigvals_f = algebra_utils.from_vector_to_diagonal_matrix(eigvals_f)
            reconstruction.append(
                ComplexMatrices.mul(eigvecs, eigvals_f, transconj_eigvecs)
            )
        return reconstruction if return_list else reconstruction[0]