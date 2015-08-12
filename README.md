# rtpipe

... or 'real-time pipeline' is a package for searching for fast transients in radio interferometer data. 
Supersedes [tpipe](http://github.com/caseyjlaw/tpipe). Uses a paradigm that defines how to break long (large) data into smaller, independent pieces with a single treatment (flagging, image gridding, calibration, etc.).

## Requirements

* Python 2.7
* Standard scientific Python stuff: numpy, scipy, matplotlib, multiprocessing
* [pwkit](http://github.com/pkgw/pwkit) (for access to CASA libraries)
* Cython (tested with 0.19.1)
* [sdmreader](http://github.com/caseyjlaw/sdmreader) (for reading SDM format data)
* [sdmpy](http://github.com/demorest/sdmreader) (for reading SDM format data)
* [pyFFTW](https://pypi.python.org/pypi/pyFFTW) (accelerated FFTs; tested with 0.92)

## Install

    > python setup.py install
