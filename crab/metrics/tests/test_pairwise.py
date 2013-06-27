import numpy as np
from ..pairwise import check_pairwise_arrays
from nose.tools import assert_true
from numpy.testing import assert_equal
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_raises
from scipy.sparse import csr_matrix

from ..pairwise import euclidean_distances, manhattan_distances
from ..pairwise import pearson_correlation, jaccard_coefficient
from ..pairwise import tanimoto_coefficient, cosine_distances
from ..pairwise import loglikehood_coefficient, sorensen_coefficient
from ..pairwise import spearman_coefficient, adjusted_cosine


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

    #Inverse Test
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = euclidean_distances(X, X, inverse=True)
    assert_array_almost_equal(D, [[1.]])

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
    assert_array_almost_equal(D, [[2.39791576, 0.], [2.39791576, 0.]])

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


def test_manhattan_distances():
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
    assert_raises(ValueError, manhattan_distances, X, Y)


def test_pearson_correlation():
    """ Check that the pairwise Pearson distances computation"""
    #Idepontent Test
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = pearson_correlation(X, X)
    assert_array_almost_equal(D, [[1.]])

    #Vector x Non Vector
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[]]
    assert_raises(ValueError, pearson_correlation, X, Y)

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
    assert_raises(ValueError, pearson_correlation, X, Y)


def test_adjusted_cosine():
    """ Check that the pairwise Pearson distances computation"""
    #Idepontent Test
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    EFV = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    D = adjusted_cosine(X, X, EFV)
    assert_array_almost_equal(D, [[1.]])

    #Vector x Non Vector
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[]]
    EFV = [[]]
    assert_raises(ValueError, adjusted_cosine, X, Y, EFV)

    #Vector A x Vector B
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5, 3.0]]
    EFV = [[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]
    D = adjusted_cosine(X, Y, EFV)
    assert_array_almost_equal(D, [[0.80952381]])

    #Vector N x 1
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5, 3.0]]
    EFV = [[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]
    D = adjusted_cosine(X, Y, EFV)
    assert_array_almost_equal(D, [[0.80952381], [0.80952381]])

    #N-Dimmensional Vectors
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5, 3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    EFV = [[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]
    D = adjusted_cosine(X, Y, EFV)
    assert_array_almost_equal(D, [[0.80952381, 1.], [0.80952381, 1.]])

    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0], [3.0, 3.5, 1.5, 5.0, 3.5, 3.0]]
    EFV = [[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]
    D = adjusted_cosine(X, X, EFV)
    assert_array_almost_equal(D, [[1., 0.80952381], [0.80952381, 1.]])

    X = [[1.0, 0.0], [1.0, 1.0]]
    Y = [[0.0, 0.0]]
    EFV = [[0.0, 0.0]]
    D = adjusted_cosine(X, Y, EFV)
    assert_array_almost_equal(D, [[np.nan], [np.nan]])

    #Test Sparse Matrices
    X = csr_matrix(X)
    Y = csr_matrix(Y)
    EFV = csr_matrix(EFV)
    assert_raises(ValueError, adjusted_cosine, X, Y, EFV)


def test_jaccard_distances():
    """ Check that the pairwise Jaccard distances computation"""
    #Idepontent Test
    X = [['a', 'b', 'c']]
    D = jaccard_coefficient(X, X)
    assert_array_almost_equal(D, [[1.]])

    #Vector x Non Vector
    X = [['a', 'b', 'c']]
    Y = [[]]
    D = jaccard_coefficient(X, Y)
    assert_array_almost_equal(D, [[0.]])

    #Vector A x Vector B
    X = [[1, 2, 3, 4]]
    Y = [[2, 3]]
    D = jaccard_coefficient(X, Y)
    assert_array_almost_equal(D, [[0.5]])

    #BUG FIX: How to fix for multi-dimm arrays

    #Vector N x 1
    X = [['a', 'b', 'c', 'd'], ['e', 'f', 'g']]
    Y = [['a', 'b', 'c', 'k']]
    D = jaccard_coefficient(X, Y)
    assert_array_almost_equal(D, [[0.6], [0.]])

    #N-Dimmensional Vectors
    X = [['a', 'b', 'c', 'd'], ['e', 'f', 'g']]
    Y = [['a', 'b', 'c', 'd'], ['e', 'f', 'g']]
    D = jaccard_coefficient(X, Y)
    assert_array_almost_equal(D, [[1., 0.], [0., 1.]])

    X = [[0, 1], [1, 2]]
    D = jaccard_coefficient(X, X)
    assert_array_almost_equal(D, [[1., 0.33333333], [0.33333333, 1.]])

    X = [[0, 1], [1, 2]]
    Y = [[0, 3]]
    D = jaccard_coefficient(X, Y)
    assert_array_almost_equal(D, [[0.33333333], [0.]])

    #Test Sparse Matrices
    X = csr_matrix(X)
    Y = csr_matrix(Y)
    assert_raises(ValueError, jaccard_coefficient, X, Y)


def test_tanimoto_distances():
    """ Check that the pairwise Tanimoto distances computation"""
    #Idepontent Test
    X = [['a', 'b', 'c']]
    D = tanimoto_coefficient(X, X)
    assert_array_almost_equal(D, [[1.]])

    #Vector x Non Vector
    X = [['a', 'b', 'c']]
    Y = [[]]
    D = tanimoto_coefficient(X, Y)
    assert_array_almost_equal(D, [[0.]])

    #Vector A x Vector B
    X = [[1, 2, 3, 4]]
    Y = [[2, 3]]
    D = tanimoto_coefficient(X, Y)
    assert_array_almost_equal(D, [[0.5]])

    #BUG FIX: How to fix for multi-dimm arrays

    #Vector N x 1
    X = [['a', 'b', 'c', 'd'], ['e', 'f', 'g']]
    Y = [['a', 'b', 'c', 'k']]
    D = tanimoto_coefficient(X, Y)
    assert_array_almost_equal(D, [[0.6], [0.]])

    #N-Dimmensional Vectors
    X = [['a', 'b', 'c', 'd'], ['e', 'f', 'g']]
    Y = [['a', 'b', 'c', 'd'], ['e', 'f', 'g']]
    D = tanimoto_coefficient(X, Y)
    assert_array_almost_equal(D, [[1., 0.], [0., 1.]])

    X = [[0, 1], [1, 2]]
    D = tanimoto_coefficient(X, X)
    assert_array_almost_equal(D, [[1., 0.33333333], [0.33333333, 1.]])

    X = [[0, 1], [1, 2]]
    Y = [[0, 3]]
    D = tanimoto_coefficient(X, Y)
    assert_array_almost_equal(D, [[0.3333333], [0.]])

    #Test Sparse Matrices
    X = csr_matrix(X)
    Y = csr_matrix(Y)
    assert_raises(ValueError, tanimoto_coefficient, X, Y)


def test_cosine_distances():
    """ Check that the pairwise Cosine distances computation"""
    #Idepontent Test
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = cosine_distances(X, X)
    assert_array_almost_equal(D, [[1.]])
    #Vector x Non Vector
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[]]
    assert_raises(ValueError, cosine_distances, X, Y)
    #Vector A x Vector B
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5, 3.0]]
    D = cosine_distances(X, Y)
    assert_array_almost_equal(D, [[0.960646301]])
    #Vector N x 1
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5, 3.0]]
    D = cosine_distances(X, Y)
    assert_array_almost_equal(D, [[0.960646301], [0.960646301]])

    #N-Dimmensional Vectors
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5, 3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = cosine_distances(X, Y)
    assert_array_almost_equal(D, [[0.960646301, 1.], [0.960646301, 1.]])

    X = [[0, 1], [1, 1]]
    D = cosine_distances(X, X)
    assert_array_almost_equal(D, [[1., 0.70710678], [0.70710678, 1.]])

    X = [[0, 1], [1, 1]]
    Y = [[0, 0]]
    D = cosine_distances(X, Y)
    assert_array_almost_equal(D, [[np.nan], [np.nan]])

    #Test Sparse Matrices
    X = csr_matrix(X)
    Y = csr_matrix(Y)
    assert_raises(ValueError, cosine_distances, X, Y)


def test_loglikehood_distances():
    """ Check that the pairwise LogLikehood distances computation"""
    #Idepontent Test
    X = [['a', 'b', 'c']]
    n_items = 3
    D = loglikehood_coefficient(n_items, X, X)
    assert_array_almost_equal(D, [[1.]])

    #Vector x Non Vector
    X = [['a', 'b', 'c']]
    Y = [[]]
    n_items = 3
    D = loglikehood_coefficient(n_items, X, Y)
    assert_array_almost_equal(D, [[0.]])

    #Vector A x Vector B
    X = [[1, 2, 3, 4]]
    Y = [[2, 3]]
    n_items = 4
    D = loglikehood_coefficient(n_items, X, Y)
    assert_array_almost_equal(D, [[0.]])

    #BUG FIX: How to fix for multi-dimm arrays

    #Vector N x 1
    X = [['a', 'b', 'c', 'd'], ['e', 'f', 'g', 'h']]
    Y = [['a', 'b', 'c', 'k']]
    n_items = 8
    D = loglikehood_coefficient(n_items, X, Y)
    assert_array_almost_equal(D, [[0.67668852], [0.]])

    #N-Dimmensional Vectors
    X = [['a', 'b', 'c', 'd'], ['e', 'f', 'g', 'h']]
    Y = [['a', 'b', 'c', 'd'], ['e', 'f', 'g', 'h']]
    n_items = 7
    D = loglikehood_coefficient(n_items, X, Y)
    assert_array_almost_equal(D, [[1., 0.], [0., 1.]])


def test_sorensen_distances():
    """ Check that the pairwise Sorensen distances computation"""
    #Idepontent Test
    X = [['a', 'b', 'c']]
    D = sorensen_coefficient(X, X)
    assert_array_almost_equal(D, [[1.]])

    #Vector x Non Vector
    X = [['a', 'b', 'c']]
    Y = [[]]
    D = sorensen_coefficient(X, Y)
    assert_array_almost_equal(D, [[0.]])

    #Vector A x Vector B
    X = [[1, 2, 3, 4]]
    Y = [[2, 3]]
    D = sorensen_coefficient(X, Y)
    assert_array_almost_equal(D, [[0.666666]])

    #BUG FIX: How to fix for multi-dimm arrays

    #Vector N x 1
    X = [['a', 'b', 'c', 'd'], ['e', 'f', 'g']]
    Y = [['a', 'b', 'c', 'k']]
    D = sorensen_coefficient(X, Y)
    assert_array_almost_equal(D, [[0.75], [0.]])

    #N-Dimmensional Vectors
    X = [['a', 'b', 'c', 'd'], ['e', 'f', 'g']]
    Y = [['a', 'b', 'c', 'd'], ['e', 'f', 'g']]
    D = sorensen_coefficient(X, Y)
    assert_array_almost_equal(D, [[1., 0.], [0., 1.]])

    X = [[0, 1], [1, 2]]
    D = sorensen_coefficient(X, X)
    assert_array_almost_equal(D, [[1., 0.5], [0.5, 1.]])

    X = [[0, 1], [1, 2]]
    Y = [[0, 0]]
    D = sorensen_coefficient(X, Y)
    assert_array_almost_equal(D, [[0.5], [0.]])


def test_spearman_distances():
    """ Check that the pairwise Spearman distances computation"""
    #Idepontent Test
    X = [[('a', 2.5), ('b', 3.5), ('c', 3.0), ('d', 3.5),
          ('e', 2.5), ('f', 3.0)]]
    D = spearman_coefficient(X, X)
    assert_array_almost_equal(D, [[1.]])

    #Vector x Non Vector
    X = [[('a', 2.5), ('b', 3.5), ('c', 3.0), ('d', 3.5),
          ('e', 2.5), ('f', 3.0)]]
    Y = [[]]
    assert_raises(ValueError, spearman_coefficient, X, Y)

    #Vector A x Vector B
    X = [[('a', 2.5), ('b', 3.5), ('c', 3.0), ('d', 3.5),
          ('e', 2.5), ('f', 3.0)]]
    Y = [[('a', 3.0), ('b', 3.5), ('c', 1.5), ('d', 5.0),
          ('e', 3.5), ('f', 3.0)]]
    D = spearman_coefficient(X, Y)
    assert_array_almost_equal(D, [[0.5428571428]])

    #Vector N x 1
    X = [[('a', 2.5), ('b', 3.5), ('c', 3.0), ('d', 3.5)],
         [('e', 2.5), ('f', 3.0), ('g', 2.5), ('h', 4.0)]]
    Y = [[('a', 2.5), ('b', 3.5), ('c', 3.0), ('k', 3.5)]]
    D = spearman_coefficient(X, Y)
    assert_array_almost_equal(D, [[1.], [0.]])

    #N-Dimmensional Vectors
    X = [[('a', 2.5), ('b', 3.5), ('c', 3.0), ('d', 3.5)],
         [('e', 2.5), ('f', 3.0), ('g', 2.5), ('h', 4.0)]]
    Y = [[('a', 2.5), ('b', 3.5), ('c', 3.0), ('d', 3.5)],
         [('e', 2.5), ('f', 3.0), ('g', 2.5), ('h', 4.0)]]
    D = spearman_coefficient(X, Y)
    assert_array_almost_equal(D, [[1., 0.], [0., 1.]])
