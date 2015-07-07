# rtpipe

... or 'real-time pipeline' is a package for searching for fast transients in radio interferometer data. 
Supersedes `tpipe` (http://github.com/caseyjlaw/tpipe) by defining algorithm to break long (large) data into smaller, independent pieces.

## Requirements

* Python 2.7 recommended
* Standard scientific Python stuff: numpy, scipy, matplotlib, multiprocessing
* pwkit (to build casapy-free CASA; http://github.com/pkgw/pwkit)
* Cython (works with 0.19.1)
* To work with SDM data:
** sdmreader (http://github.com/caseyjlaw/sdmreader)
** sdmpy (http://github.com/demorest/sdmreader)
* pyFFTW (accelerated FFTs; works with 0.9.2; https://pypi.python.org/pypi/pyFFTW)