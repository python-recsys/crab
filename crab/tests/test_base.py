
# Author: Marcel Caraciolo
# License: BSD

from numpy.testing import assert_equal
from nose.tools import assert_true
from nose.tools import assert_raises

from crab.base import BaseEstimator


#############################################################################
# A few test classes
class MyRecommender(BaseEstimator):

    def __init__(self, model=None, with_preference=False):
        self.model = model
        self.with_preference = with_preference


class KRecommender(BaseEstimator):
    def __init__(self, c=None, d=None):
        self.c = c
        self.d = d


class TRecommender(BaseEstimator):
    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b


class BuggyRecommender(BaseEstimator):
    " A buggy recommender that does not set its parameters right. "

    def __init__(self, a=None):
        self.a = 1


class NoRecommender(object):
    def __init__(self):
        pass

    def estimate_preference(self, user_id, item_id, **params):
        return self

    def recommend(self, user_id, how_many, **params):
        return None


class VargRecommender(BaseEstimator):
    """Crab recommenders shouldn't have vargs."""
    def __init__(self, *vargs):
        pass


def test_repr():
    """Smoke test the repr of the base estimator."""
    my_recommender = MyRecommender()
    repr(my_recommender)
    test = TRecommender(KRecommender(), KRecommender())
    assert_equal(
        repr(test),
        "TRecommender(a=KRecommender(c=None, d=None), " +
         "b=KRecommender(c=None, d=None))"
    )

    some_est = TRecommender(a=["long_params"] * 1000)
    assert_equal(len(repr(some_est)), 432)


def test_str():
    """Smoke test the str of the base estimator"""
    my_estimator = MyRecommender()
    str(my_estimator)


def test_get_params():
    test = TRecommender(KRecommender(), KRecommender())

    assert_true('a__d' in test.get_params(deep=True))
    assert_true('a__d' not in test.get_params(deep=False))

    test.set_params(a__d=2)
    assert_true(test.a.d == 2)
    assert_raises(ValueError, test.set_params, a__a=2)
