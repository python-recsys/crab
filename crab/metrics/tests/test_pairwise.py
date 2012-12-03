import numpy as np
from ..pairwise import check_pairwise_arrays
from nose.tools import assert_true
from numpy.testing import assert_equal
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_raises
from scipy.sparse import csr_matrix

from ..pairwise import euclidean_distances, pearson_correlation
from ..pairwise import manhattan_distances


def test_check_dense_matrices():
    """ Ensure that pairwise array check works for dense matrices."""
    # Check that if XB is None, XB is returned as reference to XA
    XA = np.resize(np.arange(40), (5, 8))
    XA_checked, XB_checked = check_pairwise_arrays(XA, None)
    assert_true(XA_checked is XB_checked)
    assert_equal(XA, XA_checked)


def test_check_XB_returned():
    """ Ensure that if XA and XB are given correctly, they return as equal."""
    # Check that if XB is not None, it is returned equal.
    # Note that the second dimension of XB is the same as XA.
    XA = np.resize(np.arange(40), (5, 8))
    XB = np.resize(np.arange(32), (4, 8))
    XA_checked, XB_checked = check_pairwise_arrays(XA, XB)
    assert_equal(XA, XA_checked)
    assert_equal(XB, XB_checked)


def test_check_different_dimensions():
    """ Ensure an error is raised if the dimensions are different. """
    XA = np.resize(np.arange(45), (5, 9))
    XB = np.resize(np.arange(32), (4, 8))
    assert_raises(ValueError, check_pairwise_arrays, XA, XB)


def test_check_invalid_dimensions():
    """ Ensure an error is raised on 1D input arrays. """
    XA = np.arange(45)
    XB = np.resize(np.arange(32), (4, 8))
    assert_raises(ValueError, check_pairwise_arrays, XA, XB)
    XA = np.resize(np.arange(45), (5, 9))
    XB = np.arange(32)
    assert_raises(ValueError, check_pairwise_arrays, XA, XB)


def test_check_sparse_arrays():
    """ Ensures that checks return valid sparse matrices. """
    rng = np.random.RandomState(0)
    XA = rng.random_sample((5, 4))
    XA_sparse = csr_matrix(XA)
    XB = rng.random_sample((5, 4))
    XB_sparse = csr_matrix(XB)
    XA_checked, XB_checked = check_pairwise_arrays(XA_sparse, XB_sparse)
    assert_equal(XA_sparse, XA_checked)
    assert_equal(XB_sparse, XB_checked)


def tuplify(X):
    """ Turns a numpy matrix (any n-dimensional array) into tuples."""
    s = X.shape
    if len(s) > 1:
        # Tuplify each sub-array in the input.
        return tuple(tuplify(row) for row in X)
    else:
        # Single dimension input, just return tuple of contents.
        return tuple(r for r in X)


def test_check_tuple_input():
    """ Ensures that checks return valid tuples. """
    rng = np.random.RandomState(0)
    XA = rng.random_sample((5, 4))
    XA_tuples = tuplify(XA)
    XB = rng.random_sample((5, 4))
    XB_tuples = tuplify(XB)
    XA_checked, XB_checked = check_pairwise_arrays(XA_tuples, XB_tuples)
    assert_equal(XA_tuples, XA_checked)
    assert_equal(XB_tuples, XB_checked)


def test_euclidean_distances():
    """Check that the pairwise euclidian distances computation"""
    #Idepontent Test
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = euclidean_distances(X, X)
    assert_array_almost_equal(D, [[0.]])

    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = euclidean_distances(X, X, inverse=False)
    assert_array_almost_equal(D, [[0.]])

    #Vector x Non Vector
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[]]
    assert_raises(ValueError, euclidean_distances, X, Y)

    #Vector A x Vector B
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5, 3.0]]
    D = euclidean_distances(X, Y)
    assert_array_almost_equal(D, [[2.39791576]])

    #Vector N x 1
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5, 3.0]]
    D = euclidean_distances(X, Y)
    assert_array_almost_equal(D, [[2.39791576], [2.39791576]])

    #N-Dimmensional Vectors
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5, 3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = euclidean_distances(X, Y)
    assert_array_almost_equal(D, [[2.39791576, 0.], [2.39791576,  0.]])

    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0], [3.0, 3.5, 1.5, 5.0, 3.5, 3.0]]
    D = euclidean_distances(X, X)
    assert_array_almost_equal(D, [[0., 2.39791576], [2.39791576, 0.]])

    X = [[1.0, 0.0], [1.0, 1.0]]
    Y = [[0.0, 0.0]]
    D = euclidean_distances(X, Y)
    assert_array_almost_equal(D, [[1.], [1.41421356]])

    #Test Sparse Matrices
    X = csr_matrix(X)
    Y = csr_matrix(Y)
    D = euclidean_distances(X, Y)
    assert_array_almost_equal(D, [[1.], [1.41421356]])


def test_pearson_correlation():
    """ Check that the pairwise Pearson distances computation"""
    #Idepontent Test
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = pearson_correlation(X, X)
    assert_array_almost_equal(D, [[1.]])

    #Vector x Non Vector
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[]]
    assert_raises(Exception, pearson_correlation, X, Y)

    #Vector A x Vector B
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5, 3.0]]
    D = pearson_correlation(X, Y)
    assert_array_almost_equal(D, [[0.3960590]])

    #Vector N x 1
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5, 3.0]]
    D = pearson_correlation(X, Y)
    assert_array_almost_equal(D, [[0.3960590], [0.3960590]])

    #N-Dimmensional Vectors
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5, 3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = pearson_correlation(X, Y)
    assert_array_almost_equal(D, [[0.3960590, 1.], [0.3960590, 1.]])

    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0], [3.0, 3.5, 1.5, 5.0, 3.5, 3.0]]
    D = pearson_correlation(X, X)
    assert_array_almost_equal(D, [[1., 0.39605902], [0.39605902, 1.]])

    X = [[1.0, 0.0], [1.0, 1.0]]
    Y = [[0.0, 0.0]]
    D = pearson_correlation(X, Y)
    assert_array_almost_equal(D, [[np.nan], [np.nan]])

    #Test Sparse Matrices
    X = csr_matrix(X)
    Y = csr_matrix(Y)
    #D = pearson_correlation(X, Y)
    #assert_array_almost_equal(D, [[1.], [1.41421356]])


def test_manthattan_distances():
    """ Check that the pairwise Manhattan distances computation"""
    #Idepontent Test
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = manhattan_distances(X, X)
    assert_array_almost_equal(D, [[1.]])

    #Vector x Non Vector
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[]]
    assert_raises(ValueError, manhattan_distances, X, Y)

    #Vector A x Vector B
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5, 3.0]]
    D = manhattan_distances(X, Y)
    assert_array_almost_equal(D, [[0.25]])

    #BUG FIX: How to fix for multi-dimm arrays

    #Vector N x 1
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5, 3.0]]
    D = manhattan_distances(X, Y)
    assert_array_almost_equal(D, [[0.25], [0.25]])

    #N-Dimmensional Vectors
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = manhattan_distances(X, Y)
    assert_array_almost_equal(D, [[1., 1.], [1., 1.]])

    X = [[0, 1], [1, 1]]
    D = manhattan_distances(X, X)
    assert_array_almost_equal(D, [[1., 0.5], [0.5, 1.]])

    X = [[0, 1], [1, 1]]
    Y = [[0, 0]]
    D = manhattan_distances(X, Y)
    assert_array_almost_equal(D, [[0.5], [0.]])

   #Test Sparse Matrices
    X = csr_matrix(X)
    Y = csr_matrix(Y)
    #D = manhattan_distances(X, Y)
    #assert_array_almost_equal(D, [[1.], [1.41421356]])
