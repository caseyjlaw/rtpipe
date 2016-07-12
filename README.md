rtpipe
==

rtpipe (as in 'real-time pipeline') is a Python package for searching visibility data from radio interferometer data for fast (subsecond) transients. In contrast to traditional time-domain techniques used on large single-dish telescopes or "phased" interferometers, fast-sampled visibilities can precisely localize sources anywhere in the entire field of view. This package supersedes [tpipe](http://github.com/caseyjlaw/tpipe) and uses a paradigm that defines how to break long (large) data into smaller, independent pieces with a single treatment (flagging, image gridding, calibration, etc.).

For a quick exploration of `rtpipe`, run this Jupyter notebook with binder: [![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/caseyjlaw/reproducing-fast-imaging-rrats).

Requirements
---

* Python 2.7
* Standard scientific Python stuff: numpy, scipy, matplotlib, multiprocessing, bokeh
* [pwkit](http://github.com/pkgw/pwkit) (for access to CASA libraries)
* Cython (tested with 0.19.1)
* [sdmpy](http://github.com/caseyjlaw/sdmpy) (for reading SDM format data)
* [pyFFTW](https://pypi.python.org/pypi/pyFFTW) (accelerated FFTs; tested with 0.92)

Install
---

If you have [anaconda](https://www.continuum.io/downloads), you can install `rtpipe` like this:

    conda install -c conda-forge numpy scipy jupyter bokeh cython matplotlib 
    conda install -c pkgw pwkit casa-tools casa-python casa-data
    pip install rtpipe activegit

Alternate Install
-----

Alternatively, you can install by building pwkit yourself to access CASA libraries. This is a bit tricky, but can be done by installing [CASA](https://casa.nrao.edu/) and setting environment variables to see its libraries:

    setenv CASA_HOME /home/casa/packages/RHEL6/stable/current
    setenv PYTHONPATH ${PYTHONPATH}:${CASA_HOME}/lib/python2.7
    setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${CASA_HOME}/lib
    setenv CASAPATH "${CASA_HOME} linux local ${HOST}"

The key is being sure that the python interpreter is binary-compatible with that used to build CASA. Note that this approach has not been tested recently and there are new pieces (e.g., bokeh) that are not explicit dependencies in the `rtpipe` build. And, really, you should be using the anaconda installer anyway!

Contributors
---
* [Casey Law](http://www.twitter.com/caseyjlaw)
* Peter Williams (CASA libraries and general advice)
* Paul Demorest (sdmpy)

