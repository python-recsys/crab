"""
The :mod:`crab.metrics.pairwise` submodule implements utilities to evaluate
pairwise distances or affinity of sets of samples.

This module contains distance metrics. A brief summary is
given below.

Distance metrics are a function d(a, b) such that d(a, b) < d(a, c) if objects
a and b are considered "more similar" to objects a and c. Two objects exactly
alike would have a distance of zero.
One of the most popular examples is Euclidean distance.
To be a 'true' metric, it must obey the following four conditions::

    1. d(a, b) >= 0, for all a and b
    2. d(a, b) == 0, if and only if a = b, positive definiteness
    3. d(a, b) == d(b, a), symmetry
    4. d(a, c) <= d(a, b) + d(b, c), the triangle inequality


"""
