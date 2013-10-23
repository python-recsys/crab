**************************
Google Summer of Code 2012
**************************

The `Google Summer of Code <http://www.google-melange.com/gsoc/homepage/google/gsoc2012>`__ is a program that offers students stipends to write code for open source projects.

Below are some suggested projects for students interested in contributing to the *pandas* project for the summer of 2012. Students are also encouraged to propose their own projects of interest! To get involved, students should review the `developer page <developers.html>`_ and email the `developer mailing list <http://groups.google.com/group/pystatsmodels>`__ to contact a mentor.

These projects will enable students to hone their abilities in Python, Cython, and C, while becoming familiar with an essential tool for Data Science used in multiple industries. 

Thanks to Arc Riley and the `Python Software Foundation <http://www.python.org/psf/>`__ for coordinating this effort.


Panel Indexing
~~~~~~~~~~~~~~

Description
-----------
In the pandas data analysis library for Python, the Panel (3D) data objects are missing several important indexing operations, foremost of which is proper hierarchical indexing, which are fully implemented in the lower-dimensional data objects, such as DataFrame (2D) and Series (1D). These indexing operations are described in the following pandas documentation:

http://pandas.pydata.org/pandas-docs/stable/indexing.html

The Panel indexing operations are necessary for API consistency. The developer will gain exposure to the internals of the pandas data analysis library, and will see how Cython and C code are used in critical code paths to optimize performance.

Expected Results
----------------
- Implement missing Panel (3D) indexing features, including hierarchical indexing
- Comprehensive test suite for new indexing features

Knowledge Prerequisite
----------------------
Python [expert], Cython [basic], C [intermediate]

Mentor
------
Adam Klein

Extending dtype support in pandas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

Description
-----------

In the pandas data analysis library for Python, all data elements are stored as one of four basic data types: 64-bit integer, 64-bit floating point, 8-bit boolean, and python object pointer. Support for additional numpy dtypes (such as int8, int16, float8, etc.) is important for several reasons, one of which is to reduce memory footprint where extra space is unnecessary; another reason is to lay the groundwork for a "structured array" back-end for the pandas data objects. This latter implementation is an important step toward supporting memory-mapped objects for larger-than-memory data sets in pandas. The developer will become familiar with the internals of pandas as well as the C, Cython, and numpy extension code that drives pandas array-based processing.

Expected Results
----------------
- extend Panel, DataFrame, and Series functionality to work on additional numpy dtypes
- comprehensive test additions to test suite to verify functionality

Knowledge Prerequisite
----------------------
Python [expert], C [intermediate], Cython [intermediate], Numpy [intermediate]

Mentor
------
Wes McKinney

Plots in pandas
~~~~~~~~~~~~~~~

Description
-----------

The pandas data analysis library for Python provides 1D (Series), 2D (DataFrame), and 3D (Panel) data objects with which to do interactive analysis. One of the weaker areas of the library is visualization.  Many additional plots beyond the already-included histogram, line, and box plots should be added to the core library, which will help generally with pandas data visualization through matplotlib, the standard graphing library for scientific Python. The developer will become familiar with the internals of pandas, as well as an expert in plot generation in matplotlib.

http://pandas.pydata.org/pandas-docs/stable/visualization.html

Expected Results
----------------

- additional plots and plot options to visualize data stored in pandas objects
- comprehensive test additions to test suite to verify functionality

Knowledge Prerequisite
----------------------

Python [expert], C [basic], Cython [basic], Matplotlib [basic]

Mentor
------
Adam Klein

pandas GUI
~~~~~~~~~~

Description
-----------

The pandas data analysis library for Python provides 1D (Series), 2D (DataFrame), and 3D (Panel) data objects with which to do interactive analysis. Currently the only way to interact with these objects is through the API. This project proposes to add a simple Qt or Tk GUI with which to view and manipulate these objects. For instance, a 2D (DataFrame) viewer would provide a spreadsheet-based GUI or widget, while a 3D (Panel) viewer could provide similar views as projections of the 3D object. This would provide an alternative way of interacting with data loaded into the pandas data objects.

Expected Results
----------------

- a simple GUI with which to view and edit pandas data objects
- comprehensive test additions to test suite to verify functionality

Knowledge Prerequisite
----------------------
Python [expert], Tk or Qt [intermediate]

Mentor
------
Chang She