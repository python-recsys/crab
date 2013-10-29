"""Utilities for input validation"""

# Taken from scikit-learn.
# Authors: Olivier Grisel
#          Gael Varoquaux
#          Andreas Mueller
#          Lars Buitinck


from scipy import sparse
import numpy as np
from .fixes import safe_copy


def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    if X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum()) and not np.isfinite(X).all():
        raise ValueError("Array contains NaN or infinity.")


def assert_all_finite(X):
    """Throw a ValueError if X contains NaN or infinity.

    Input MUST be an np.ndarray instance or a scipy.sparse matrix."""

    # First try an O(n) time, O(1) space solution for the common case that
    # there everything is finite; fall back to O(n) space np.isfinite to
    # prevent false positives from overflow in sum method.
    _assert_all_finite(X.data if sparse.issparse(X) else X)


def safe_asarray(X, dtype=None, order=None):
    """Convert X to an array or sparse matrix.

    Prevents copying X when possible; sparse matrices are passed through."""
    if sparse.issparse(X):
        assert_all_finite(X.data)
    else:
        X = np.asarray(X, dtype, order)
        assert_all_finite(X)
    return X


def array2d(X, dtype=None, order=None, copy=False):
    """Returns at least 2-d array with data from X"""
    if sparse.issparse(X):
        raise TypeError('A sparse matrix was passed, but dense data '
                        'is required. Use X.toarray() to convert to dense.')
    X_2d = np.asarray(np.atleast_2d(X), dtype=dtype, order=order)
    _assert_all_finite(X_2d)
    if X is X_2d and copy:
        X_2d = safe_copy(X_2d)
    return X_2d


def _atleast2d_or_sparse(X, dtype, order, copy, sparse_class, convmethod):
    if sparse.issparse(X):
        # Note: order is ignored because CSR matrices hold data in 1-d arrays
        if dtype is None or X.dtype == dtype:
            X = getattr(X, convmethod)()
        else:
            X = sparse_class(X, dtype=dtype)
        _assert_all_finite(X.data)
    else:
        X = array2d(X, dtype=dtype, order=order, copy=copy)
        _assert_all_finite(X)
    return X


def atleast2d_or_csc(X, dtype=None, order=None, copy=False):
    """Like numpy.atleast_2d, but converts sparse matrices to CSC format.

    Also, converts np.matrix to np.ndarray.
    """
    return _atleast2d_or_sparse(X, dtype, order, copy, sparse.csc_matrix,
                                "tocsc")


def atleast2d_or_csr(X, dtype=None, order=None, copy=False):
    """Like numpy.atleast_2d, but converts sparse matrices to CSR format

    Also, converts np.matrix to np.ndarray.
    """
    return _atleast2d_or_sparse(X, dtype, order, copy, sparse.csr_matrix,
                                "tocsr")
