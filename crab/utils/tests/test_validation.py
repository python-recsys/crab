"""Tests for input validation functions"""
import numpy as np
from nose.tools import assert_false, assert_true
import scipy.sparse as sp

from crab.utils import atleast2d_or_csr, atleast2d_or_csc, safe_asarray


def test_np_matrix():
    """Confirm that input validation code does not return np.matrix"""
    X = np.arange(12).reshape(3, 4)

    assert_false(isinstance(atleast2d_or_csr(X), np.matrix))
    assert_false(isinstance(atleast2d_or_csr(np.matrix(X)), np.matrix))
    assert_false(isinstance(atleast2d_or_csr(sp.csc_matrix(X)), np.matrix))

    assert_false(isinstance(atleast2d_or_csc(X), np.matrix))
    assert_false(isinstance(atleast2d_or_csc(np.matrix(X)), np.matrix))
    assert_false(isinstance(atleast2d_or_csc(sp.csr_matrix(X)), np.matrix))

    assert_false(isinstance(safe_asarray(X), np.matrix))
    assert_false(isinstance(safe_asarray(np.matrix(X)), np.matrix))
    assert_false(isinstance(safe_asarray(sp.lil_matrix(X)), np.matrix))

    assert_true(atleast2d_or_csr(X, copy=False) is X)
    assert_false(atleast2d_or_csr(X, copy=True) is X)
    assert_true(atleast2d_or_csc(X, copy=False) is X)
    assert_false(atleast2d_or_csc(X, copy=True) is X)
