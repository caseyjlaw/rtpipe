from __future__ import print_function, division, absolute_import

import multiprocessing as mp
import multiprocessing.sharedctypes as mps
from contextlib import closing
import numpy as np
from numba import jit
from numba.types import boolean, complex64, float32
import rtpipe.parsecal as pc
import logging
import pyfftw

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def pipeline_dataprep(d, segment):
    """ Read and prepare data for search. Use numpy arrays. """

    d['segment'] = segment
    data = readsegment(datashape(d), mode='numpy')
    data *= pc.getgains(d, segment)

    chanranges = [xrange(*d['spw_chanr_select'][ss]) for ss in d['spw']]
    pols = range(d['npol'])
    flags = calcflags(data, d['flaglist'], chanranges, pols)
    data *= flags

    data -= calcmeant(data)

    return data


def pipeline_dataprep2(d):
    """ Read and prepare data for search. Use shared memory for multiprocessing """

    data_mem = mps.Array(mps.ctypes.c_float, datasize(d)*2)
    u_mem = mps.Array(mps.ctypes.c_float, d['nbl'])
    v_mem = mps.Array(mps.ctypes.c_float, d['nbl'])
    w_mem = mps.Array(mps.ctypes.c_float, d['nbl'])
    data = numpyview(data_mem, 'complex64', datashape(d), raw=False)
    u = numpyview(u_mem, 'float32', d['nbl'], raw=False)
    v = numpyview(v_mem, 'float32', d['nbl'], raw=False)
    w = numpyview(w_mem, 'float32', d['nbl'], raw=False)

    data[:] = readsegment(datashape(d), mode='mem')
    # start up pool, etc...


def readsegment(shape=(100,351,256,2), mode='numpy'):
    """ Mock data reader. Can return shared data_mem or data (numpy array) """

    if mode == 'mem':
        data_mem = mps.Array(mps.ctypes.c_float, 2*shape[0]*shape[1]*shape[2]*shape[3])
        data = numpyview(data_mem, 'complex64', shape)
        data.real[:] = np.random.normal(size=shape)
        data.imag[:] = np.random.normal(size=shape)
    else:
        data = np.zeros(shape=shape, dtype=np.complex64)
        data.real = np.random.normal(size=shape)
        data.imag = np.random.normal(size=shape)

    return data


def calcflags(data, flaglist, chanranges, pols):
    """ Calculates flags on data array given list of flags """

    flags = np.ones(shape=data.shape, dtype=boolean)

    for flagdef in flaglist:
        mode, sigma, conv = flagdef  # may want to redefine standard

        if mode == 'blstd':
            flags *= calcflags_blstd(data, chanranges, pols)
        elif mode == 'badchtslide':
            flags *= calcflags_badchtslide(data, chanranges, pols)
        elif mode == 'badap':
            logger.warn('Flagging mode {0} not yet implemented'.format(mode))
        else:
            logger.warn('Flagging mode {0} not available'.format(mode))

    return flags


@jit(nopython=True)
def calcflags_blstd(data, chanranges, pols):
    """ """

    for chans in chanranges:
        for pol in pols:
            blstd = blstd3d(data[:, :, chans, pol])
            blstdmed = np.median(blstd.flatten())
            blstdstd = 1.4826*calcmad(blstd.flatten())
            flags[:, :, chans, pol] = (blstd < blstdmed + sigma*blstdstd)[:,None,:]

    return flags


@jit(nopython=True)
def calcflags_badchtslide(data, chanranges, pols):
    """ """

    for chans in chanranges:
        for pol in pols:
            win = 20
            meanamp = np.abs(data[:,:,chans,pol]).mean(axis=1)

            meanspec = meanamp.mean(axis=0)
            meddevspec = np.zeros_like(meanspec)
            for ch in range(len(meanspec)):
                rr = range(max(0, ch-win/2), min(len(meanspec), ch+win/2))
                rr.remove(ch)
                meddevspec[ch] = meanspec[ch] - np.median(meanspec[rr])
            flags[:, :, chans, pol] = (meddevspec < sigma*meddevspec.std())[None, None, :]

            meanlc = meanamp.mean(axis=1)
            meddevlc = np.zeros_like(meanlc)
            for t in range(len(meanlc)):
                rr = range(max(0, t-win/2), min(len(meanlc), t+win/2))
                rr.remove(t)
                meddevlc[t] = meanlc[t] - np.median(meanlc[rr])
            flags[:, :, chans, pol] = (meddevlc < sigma*meddevlc.std())[:, None, None]

    return flags


@jit(nopython=True)
def std1d(data):
    """ Takes 1d input data and measures std while ignoring zeros"""

    mn = 0
    nonzero = 0
    for da in data:
        mn += da
        if da != 0j:
            nonzero += 1

    if nonzero:
        mn /= nonzero
    else:
        return 0j

    sq = 0
    for da in data:
        sq += np.abs(da-mn)**2

    return np.sqrt(sq/nonzero)


@jit(nopython=True)
def blstd3d(data):
    """ Takes 3d (nints, nbl, nch) input data and measures std while ignoring zeros"""

    nints, nbl, nch = data.shape
    output = np.zeros(shape=(nints, nch), dtype=float32)

    for j in xrange(nints):
        for k in xrange(nch):
            mn = 0
            nonzero = 0
            for bl in data[j, :, k]:
                mn += bl
                if bl != 0j:
                    nonzero += 1

            if nonzero:
                mn /= nonzero
                sq = 0
                for bl in data[j, :, k]:
                    sq += np.abs(bl-mn)**2

                output[j,k] = np.sqrt(sq/nonzero)
            else:
                output[j,k] = 0.

    return output


@jit(nopython=True)
def calcmad(data):
    """ Calculate median absolute deviation of 1d array """

    return np.median(np.abs(data - np.median(data)))


@jit(nopython=True)
def calcmeant(data):
    """ Subtracts mean visibility in time, ignoring zeros """

    iterints, nbl, nchan, npol = data.shape
    meant = np.zeros(shape=(nbl, nchan, npol), dtype=data.dtype)

    for j in xrange(nbl):
        for k in xrange(nchan):
            for l in xrange(npol):
                nz = data[:,j,k,l].nonzero()[0]
                if len(nz):
                    meant[j,k,l] = data[nz,j,k,l].mean()
                else:
                    meant[j,k,l] = 0j

    return meant


def datashape(d):
    return (d['readints'], d['nbl'], d['nchan'], d['npol'])


def datasize(d):
    return long(d['readints']*d['nbl']*d['nchan']*d['npol'])


def numpyview(arr, datatype, shape):
    """ Takes mp shared mp.Array and returns numpy array with given shape. """

    return np.frombuffer(arr.get_obj(),
                         dtype=np.dtype(datatype)).view(np.dtype(datatype)).reshape(shape)


def initdata(arr_mem):
    """ Share numpy array arr via global data_mem. Must be inhereted, not passed as an argument """

    global data_mem
    data_mem = arr_mem
