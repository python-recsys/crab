**********
Developers
**********

So you want to contribute to *pandas* by offering a patch? Excellent! This is
the page you need to read. After you read the below guidelines, the first
place to contribute issues & ideas to *pandas* is the `Github Issue Tracker
<https://github.com/pydata/pandas/issues>`__. You can filter using the
`"Community" <https://github.com/pydata/pandas/issues?labels=Community&state=open>`__ label to see issues we believe are easy entry points for community
contribution. Some longer discussions occur on the #pydata channel on
irc.freenode.net, on the `user <http://groups.google.com/group/pydata>`__ or
`developer <http://mail.python.org/mailman/listinfo/pandas-dev>`__ 
mailing list.

Contributing to the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're not the developer type, contributing to the documentation is still
of huge value. You don't even have to be an expert on
*pandas* to do so! Something as simple as rewriting small passages for clarity
as you reference the docs is a simple but effective way to contribute. The
next person to read that passage will be in your debt!

Actually, there are sections of the docs that are worse off by being written
by experts. If something in the docs doesn't make sense to you, updating the
relevant section after you figure it out is a simple way to ensure it will
help the next person.

Once you have followed the steps below to download the source code from github
and set up your environment, make sure you have built the C extensions in place,
then navigate to your local pandas/docs directory and run

::

     python make.py html

It will take awhile to build the first time, but subsequent builds only process
the portions you've changed. Then just open the following file in a web
browser:

::

    pandas/docs/build/html/index.html

And you'll have the satisfaction of seeing your new and improved documentation!

The documentation is written in reStructuredText, which is almost like writing
in plain English, and built using `Sphinx <http://sphinx.pocoo.org/>`__. The
Sphinx Documentation has an excellent `introduction to reST
<http://sphinx.pocoo.org/rest.html>`__. Review the Sphinx docs to perform more
complex changes to the documentation as well.

Another way to help is to **review docstrings**. These can be edited directly
in the codebase to make the functionality easier to understand.

Step-by-step overview
~~~~~~~~~~~~~~~~~~~~~

#. Read carefully through the below guidelines on working with *pandas* code.
#. Find a bug or feature you'd like to work on.
#. Create a free account on `github <http://www.github.com>`__, where we host our version controlled source repository.
#. Set up your local development environment with git (`Instructions <http://help.github.com/set-up-git-redirect>`__).
#. Fork the `pandas repository <http://www.github.com./pydata/pandas>`__ (`Instructions <http://help.github.com/fork-a-repo/>`__).
#. Create a new working branch for your changes.
#. Make sure your patch includes test coverage and performance benchmarks!
#. `Hook up Travis-CI <http://about.travis-ci.org/docs/user/getting-started/>`__
#. Commit your changes and submit a pull request (`Instructions <http://help.github.com/send-pull-requests/>`__).

Development Roadmap
~~~~~~~~~~~~~~~~~~~

* (0.13) Improved SQL / relational database tools
* Tools for working with data sets that do not fit into memory
* (0.10) Better memory usage and performance when reading very large CSV files
* Better statistical graphics using matplotlib
* `Integration with D3.js <https://github.com/mikedewar/D3py>`__
* Better support for integer ``NA`` values
* Extend GroupBy functionality to regular ndarrays, record arrays
* ✔ ``numpy.datetime64`` integration, ``scikits.timeseries`` codebase
  integration. Substantially improved time series functionality.
* ✔ Improved PyTables (HDF5) integration
* ✔ ``NDFrame`` data structure for arbitrarily high-dimensional labeled data
* ✔ Better support for NumPy dtype hierarchy without sacrificing usability
* ✔ Add a Factor data type (in R parlance)

Code design and organization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

File Hierarchy
--------------

* ``pandas/core``: Primary data structures (Series, DataFrame, ...) and related
  tools and algorithms
* ``pandas/src``: Cython and C code for implementing fundamental algorithms
* ``pandas/io``: Input/output tools (flat files, Excel, HDF5, SQL, ...)
* ``pandas/tools``: Auxiliary data algorithms: merge and join routines,
  concatenation, pivot tables, and more.
* ``pandas/sparse``: Sparse versions of Series, DataFrame, Panel
* ``pandas/stats``: Linear and panel regression, moving window regression. Will
  likely move to ``statsmodels`` eventually
* ``pandas/util``: Utilities, development, and testing tools
* ``pandas/rpy``: RPy2 interface for connecting to R

Conventions
-----------

* `PEP8 <http://www.python.org/dev/peps/pep-0008/>`__. We recommend using the
  `flake8 <http://pypi.python.org/pypi/flake8>`__ tool for checking the style
  of your code.

.. note::

   Note that pandas is not 100% PEP8 compliant but we're working on it. If you
   could help us toward this goal, it would be very helpful.


Working with the code
~~~~~~~~~~~~~~~~~~~~~

Version Control, Git, and Github
--------------------------------

The code is hosted on `Github <https://www.github.com/pydata/pandas>`_. To
contribute you will need to sign up for a `free Github account
<https://github.com/signup/free>`_. We use `Git <http://git-scm.com/>`_ for
version control to allow many people to work together on the project.

Some great resources for learning git:

 * the `Github help pages <http://help.github.com/>`__.
 * the `NumPy's documentation <http://docs.scipy.org/doc/numpy/dev/index.html>`__.
 * Matthew Brett's `Pydagogue <http://matthew-brett.github.com/pydagogue/>`__.

Getting Started with Git
------------------------

`Github has instructions <http://help.github.com/set-up-git-redirect>`__ for installing git, setting up your SSH key, and configuring git.

Forking
-------

You will need your own fork to work on the code. Go to the `pandas project
page <https://github.com/pydata/pandas>`__ and hit the *fork* button. You will
want to clone your fork to your machine: ::

    git clone git@github.com:your-user-name/pandas.git pandas-yourname
    cd pandas-yourname
    git remote add upstream git://github.com/pydata/pandas.git

This creates the directory `pandas-yourname` and connects your repository to
the upstream (main project) pandas repository.

Creating a Branch
-----------------

You want your master branch to reflect only production-ready code, so create a
feature branch for making your changes. For example::

    git branch shiny-new-feature
    git checkout shiny-new-feature

This changes your working directory to the shiny-new-feature branch.

Making changes
--------------

Now hack away! Keep any changes in this branch specific to one bug or feature so it is clear what the branch brings to pandas.

Once you've made changes, you can see them by typing::

    git status

If you've created a new file, it is not being tracked by git. Add it by typing ::

    git add path/to/file-to-be-added.py

Doing 'git status' again should give something like ::

    # On branch shiny-new-feature
    #
    #       modified:   /relative/path/to/file-you-added.py
    #

Finally, commit your changes to your local repository with an explanatory message, such as ::

    git commit -m "Optimized such-and-such function"

Your changes are now committed in your local repository.

Pushing your changes
--------------------

When you want your changes to appear publicly on your Github page, push your
forked feature branch's commits ::

    git push origin shiny-new-feature

Here `origin` is the default name given to your remote repository on Github.
You can see the remote repositories ::

    git remote -v

If you added the upstream repository as described above you will see something
like ::

    origin  git@github.com:yourname/pandas.git (fetch)
    origin  git@github.com:yourname/pandas.git (push)
    upstream        git://github.com/pydata/pandas.git (fetch)
    upstream        git://github.com/pydata/pandas.git (push)

Now your code is on Github, but it is not yet a part of the pandas project.
Before we get there, we need to address our testing and performance
requirements for new code.

Testing
~~~~~~~

Test driven development
-----------------------

We're serious about `Test Driven Development (TDD)
<http://en.wikipedia.org/wiki/Test-driven_development>`__. Any code you
contribute must have adequate test coverage to be considered.

Like many packages, *pandas* uses the `Nose testing system
<http://somethingaboutorange.com/mrl/projects/nose/>`__ and the convenient
extensions in `numpy.testing
<http://docs.scipy.org/doc/numpy/reference/routines.testing.html>`__.

Running the test suite
----------------------

The best way to develop *pandas* is to build the C extensions in-place by
running:

::

    python setup.py build_ext --inplace

The tests can then be run directly inside your git clone (without having to
install pandas) by typing:

::

    nosetests pandas

Another very common option is to do a ``develop`` install of pandas:

::

    python setup.py develop

This makes a symbolic link that tells the Python interpreter to import pandas
from your development directory. Thus, you can always be using the development
version on your system without being inside the clone directory.

How to write a test
-------------------

The ``pandas.util.testing`` module has many special ``assert`` functions that
make it easier to make statements about whether Series or DataFrame objects are
equivalent. The easiest way to verify that your code is correct is to
explicitly construct the result you expect, then compare the actual result to
the expected correct result:

::

    def test_pivot(self):
        data = {
            'index' : ['A', 'B', 'C', 'C', 'B', 'A'],
            'columns' : ['One', 'One', 'One', 'Two', 'Two', 'Two'],
            'values' : [1., 2., 3., 3., 2., 1.]
        }

        frame = DataFrame(data)
        pivoted = frame.pivot(index='index', columns='columns', values='values')

        expected = DataFrame({
            'One' : {'A' : 1., 'B' : 2., 'C' : 3.},
            'Two' : {'A' : 1., 'B' : 2., 'C' : 3.}
        })

        assert_frame_equal(pivoted, expected)

Performance testing with vbench
-----------------------------------

We created the `vbench library <https://github.com/pydata/vbench>`__ library
to enable easy monitoring of the performance of critical pandas operations.
These benchmarks are all found in the ``pandas/vb_suite`` directory.
Interested users should simply look at the code there for the latest vbench
API as ``vbench`` is still somewhat experimental and subject to change.

Contributing your changes to pandas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, double check your code
-----------------------------

When you're ready to ask for a code review, you will file a pull request. Before you do, again make sure you've followed all the guidelines outlined in this document. You should also double check your branch changes against the branch it was based off of:

#. Navigate to your repository on Github.
#. Click on `Branches`.
#. Click on the `Compare` button for your feature branch.
#. Select the `base` and `compare` branches, if necessary. This will be `master` and `shiny-new-feature`, respectively.

Then, decide if you need to rebase
----------------------------------

If you can avoid it, don't rebase. But if there has been work in
upstream/master related to the work in your branch since you started your
patch, you may need to rebase.

A rebase replays commits from one branch on top of another branch to preserve
a linear history. Remember, your commits may have been tested against an
older version of master. If you rebase, you may introduce bugs. But if you don't rebase, the two patches may conflict with each other!

Always make a new branch before doing rebase, and make sure you `thoroughly understand rebasing <http://help.github.com/rebase/>`__ lest you invoke the wrath of the git gods.

Finally, make the pull request
------------------------------

If everything looks good you are ready to make a pull request:

#. Navigate to your repository on Github.
#. Click on the `Pull Request` button.
#. You can then click on `Commits` and `Files Changed` to make sure everything looks okay one last time.
#. Write a description of your changes in the `Preview Discussion` tab.
#. Click `Send Pull Request`.

This request then appears to the repository maintainers, and they will review
the code. If you need to make more changes, you can make them in
your branch, push them to Github, and the pull request will be automatically
updated.

Optional: delete your merged branch
-----------------------------------

Once your feature branch is accepted into upstream, you'll probably want to get rid of the branch. First, merge upstream master into your branch so git knows it is safe to delete your branch ::

    git fetch upstream
    git checkout master
    git merge upstream/master

Then you can just do::

    git branch -d shiny-new-feature

Make sure you use a lower-case -d, or else git won't warn you if your feature
branch has not actually been merged.

The branch will still exist on Github, so to delete it there do ::

    git push origin --delete shiny-new-feature
