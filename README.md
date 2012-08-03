# Crab - A Recommendation Engine library for Python

  Crab is a ﬂexible, fast recommender engine for Python that integrates classic information ﬁltering recom- 
  mendation algorithms in the world of scientiﬁc Python packages (numpy, scipy, matplotlib). The engine aims 
  to provide a rich set of components from which you can construct a customized recommender system from a 
  set of algorithms.

## Usage

  For Usage and Instructions checkout the [Crab Wiki](https://github.com/python-recsys/crab/wiki)

## History
  
  The project was started in 2010  by Marcel Caraciolo as a M.S.C related  project, and since then many people interested joined to help in the project.
  It is currently maintained by a team of volunteers, members of the Python-Recsys Labs.  

## Bugs, Feedback

  Please submit bugs you might encounter, as well Patches and Features Requests to the [Issues Tracker](https://github.com/muricoca/crab/issues) located at GitHub.

## Authors
  See the AUTHORS.rst file for a complete list of contributors.

## Contributions

  If you want to submit a patch to this project, it is AWESOME. Follow this guide:
  
  * Fork Crab
  * Make your alterations and commit
  * Create a topic branch - git checkout -b my_branch
  * Push to your branch - git push origin my_branch
  * Create a [Pull Request](http://help.github.com/pull-requests/) from your branch.
  * You just contributed to the Crab project!


Dependencies
============

The required dependencies to build the software are Python >= 2.6,
setuptools, Numpy >= 1.3, SciPy >= 0.7 and a working C/C++ compiler.
This configuration matches the Ubuntu 10.04 LTS release from April 2010.

To run the tests you will also need nose >= 0.10.


Install
=======

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

  python setup.py install --home

To install for all users on Unix/Linux::

  python setup.py build
  sudo python setup.py install


Testing
-------

After installation, you can launch the test suite from outside the
source directory (you will need to have nosetest installed)::

    python -c "import crab; crab.test()"

See web page http://python-recsys.github.com/install.html#testing
for more information.


## Wiki

Please check our [Wiki](https://github.com/python-recsys/crab/wiki "Crab Wiki") wiki, for further information on how to start developing or use Crab in your projects.

## LICENCE (BSD)

Copyright (c) 2012, Python Recsys Labs

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Muriçoca Labs nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL MURIÇOCA LABS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

