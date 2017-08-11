rtpipe
==

<a href="http://ascl.net/1706.002"><img src="https://img.shields.io/badge/ascl-1706.002-blue.svg?colorB=262255" alt="ascl:1706.002" /></a>
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/caseyjlaw/reproducing-fast-imaging-rrats).


`rtpipe` (as in 'real-time pipeline') is a Python library for searching radio interferometry data for fast (subsecond) transients. This differs from traditional time-domain techniques used on large single-dish telescopes in that visibilities also measure spatial information (i.e., make images).

To view a demonstration of `rtpipe`, see this [Jupyter notebook with FRB 121102 data](https://github.com/caseyjlaw/FRB121102/blob/master/demo_FRB121102.ipynb). To execute that notebook, you will need to install the code and download about 1 GB of data. Alternatively, you can also explore some simpler features in a Jupyter notebook using [Binder](http://mybinder.org:/repo/caseyjlaw/reproducing-fast-imaging-rrats).

`rtpipe` supersedes [tpipe](http://github.com/caseyjlaw/tpipe) by using a paradigm that defines how to break long (large) data into smaller, independent pieces with a single treatment (flagging, image gridding, calibration, etc.).

Requirements
---

* Python 2.7
* Scientific Python (e.g., supported by NumFOCUS): numpy, scipy, matplotlib, multiprocessing, bokeh
* Cython 0.19.1+
* [pyFFTW](https://pypi.python.org/pypi/pyFFTW) 0.92+
* [pwkit](http://github.com/pkgw/pwkit) (for access to CASA libraries)
* [sdmpy](http://github.com/caseyjlaw/sdmpy) (for reading SDM format data)

Install
---

If you have [anaconda](https://www.continuum.io/downloads), you can install `rtpipe` like this:

    conda install -c conda-forge numpy scipy jupyter bokeh cython matplotlib pwkit casa-tools casa-python casa-data
    pip install rtpipe

I've had one report that installing from the conda-forge channel does not pick up some packages. If so, you may also try installing like this:

    conda install -c conda-forge numpy scipy jupyter bokeh cython matplotlib
    conda install -c pkgw-forge pwkit casa-tools casa-python casa-data
    pip install rtpipe

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

Citation
----
If you use rtpipe, please support open software by citing the record on the [Astrophysics Source Code Library](ascl.net) at http://ascl.net/1706.002. In AASTeX, you can do this like so:
```
\software{..., rtpipe \citep{2017ascl.soft06002L}, ...}
```

Acknowledgements
--------
This code has been supported by the University of California Office of the President under Lab Fees Research Program Award
237863 and NSF ATI program under Grant 1611606.
