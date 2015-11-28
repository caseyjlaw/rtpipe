rtpipe
==

... or 'real-time pipeline' is a package for searching for fast transients in radio interferometer data. 
Supersedes [tpipe](http://github.com/caseyjlaw/tpipe). Uses a paradigm that defines how to break long (large) data into smaller, independent pieces with a single treatment (flagging, image gridding, calibration, etc.).

Requirements
---

* Python 2.7
* Standard scientific Python stuff: numpy, scipy, matplotlib, multiprocessing
* [pwkit](http://github.com/pkgw/pwkit) (for access to CASA libraries)
* Cython (tested with 0.19.1)
* [sdmreader](http://github.com/caseyjlaw/sdmreader) (for reading SDM format data)
* [sdmpy](http://github.com/demorest/sdmreader) (for reading SDM format data)
* [pyFFTW](https://pypi.python.org/pypi/pyFFTW) (accelerated FFTs; tested with 0.92)

Install
---
    python setup.py install

pwkit can be built similarly, but requires environment variables set like so to see libraries:

    setenv CASA_HOME /home/casa/packages/RHEL6/stable/current
    setenv PYTHONPATH ${PYTHONPATH}:${CASA_HOME}/lib/python2.7
    setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${CASA_HOME}/lib
    setenv CASAPATH "${CASA_HOME} linux local ${HOST}"

The trick here is being sure that the python interpreter is binary-compatible with that used to build CASA. pwkit recommends using conda as a more stable and portable solution than that given here.

If using conda, can also install pwkit with `conda install -c pkgw casa-python casa-data` and then simply `pip install rtpipe`!

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/caseyjlaw/docker-rtpipe)

Contributors
---
[Casey Law](http://www.twitter.com/caseyjlaw)

and others...
