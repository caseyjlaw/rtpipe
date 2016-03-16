rtpipe
==

rtpipe (as in 'real-time pipeline') is a Python package for searching visibility data from radio interferometer data for fast (subsecond) transients. In contrast to traditional time-domain techniques used on large single-dish telescopes or "phased" interferometers, fast-sampled visibilities can precisely localize sources anywhere in the entire field of view. This package supersedes [tpipe](http://github.com/caseyjlaw/tpipe) and uses a paradigm that defines how to break long (large) data into smaller, independent pieces with a single treatment (flagging, image gridding, calibration, etc.).

Requirements
---

* Python 2.7
* Standard scientific Python stuff: numpy, scipy, matplotlib, multiprocessing, bokeh
* [pwkit](http://github.com/pkgw/pwkit) (for access to CASA libraries)
* Cython (tested with 0.19.1)
* [sdmreader](http://github.com/caseyjlaw/sdmreader) (for reading SDM format data)
* [sdmpy](http://github.com/demorest/sdmreader) (for reading SDM format data)
* [pyFFTW](https://pypi.python.org/pypi/pyFFTW) (accelerated FFTs; tested with 0.92)

Install
---
    pip install rtpipe

pwkit is needed to access CASA libraries. If using conda, can also install pwkit with `conda install -c pkgw casa-python casa-data` and then simply `pip install rtpipe`. Alternatively, one can build pwkit and environment variables set like so to see libraries of an existing CASA installation:

    setenv CASA_HOME /home/casa/packages/RHEL6/stable/current
    setenv PYTHONPATH ${PYTHONPATH}:${CASA_HOME}/lib/python2.7
    setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${CASA_HOME}/lib
    setenv CASAPATH "${CASA_HOME} linux local ${HOST}"

The trick here is being sure that the python interpreter is binary-compatible with that used to build CASA.

Contributors
---
* [Casey Law](http://www.twitter.com/caseyjlaw)
* Peter Williams (CASA libraries and general advice)
* Paul Demorest (sdmpy)

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/caseyjlaw/docker-rtpipe)
