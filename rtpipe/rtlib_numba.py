from __future__ import print_function, division, absolute_import

import multiprocessing as mp
import multiprocessing.sharedctypes as mps
from contextlib import closing
import numpy as np
from numba import jit, guvectorize
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

    flags = calcflags(d, data, d['flaglist'])
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
        data = np.zeros(shape=shape, dtype="complex64")
        data.real = np.random.normal(size=shape)
        data.imag = np.random.normal(size=shape)

    return data


def calcflags(d, data, flaglist):
    """ Calculates flags on data array given list of flags """

    # define how to iterate
    chanranges = [range(*d['spw_chanr_select'][ss]) for ss in d['spw']]
    pols = range(d['npol'])

    flags = np.ones(shape=data.shape, dtype="bool") # flags accumulated here

    for flagdef in flaglist:
        mode, sigma, conv = flagdef  # may want to redefine standard

        if mode == 'blstd':
            blstd = np.zeros(shape=(data.shape[0], data.shape[2], data.shape[3]), dtype="float32")
            calcblstd(data, blstd)

            for chans in chanranges:
                for pol in pols:
                    blstdmed = np.median(blstd[:,chans,pol].flatten())
                    blstdstd = 1.4826*calcmad(blstd[:,chans,pol].flatten())
                    calcflags_blstd(flags, blstd, blstdmed, blstdstd, sigma, chans, pol)

        elif mode == 'badchslide':
            meanamp = np.abs(data).mean(axis=1) # includes zeros. move to jit?
            for pol in pols:
                meanspec = meanamp[...,pol].mean(axis=0) # includes zeros. move to jit?
                meddevspec = np.zeros_like(meanspec)
                meddev(meanspec, meddevspec)

                for chans in chanranges:
                    calcflags_badchslide(flags, meanspec, meddevspec, sigma, chans, pol)

        elif mode == 'badtslide':
            meanamp = np.abs(data).mean(axis=1) # includes zeros. move to jit?
            for chans in chanranges:
                for pol in pols:
                    meanlc = meanamp[:,chans,pol].mean(axis=1) # includes zeros. move to jit?
                    meddevlc = np.zeros_like(meanlc)
                    meddev(meanlc, meddevlc)

                    calcflags_badchslide(flags, meanlc, meddevlc, sigma, chans, pol)

        elif mode == 'badap':
            logger.warn('Flagging mode {0} not yet implemented'.format(mode))

        else:
            logger.warn('Flagging mode {0} not available'.format(mode))

    return flags


@jit(nopython=True)
def calcblstd(data, blstd):
    """ Takes 4d data and fills 3d blstd measurement ignoring zeros """

    for i in xrange(data.shape[0]):
        for j in xrange(data.shape[2]):
            for k in xrange(data.shape[3]):
                mn = 0
                nonzero = 0
                for bl in data[i, :, j, k]:
                    mn += bl
                    if bl != 0j:
                        nonzero += 1

                if nonzero:
                    mn /= nonzero

                sq = 0
                for bl in data[i, :, j, k]:
                    sq += np.abs(bl - mn)**2

                if nonzero:
                    blstd[i,j,k] = np.sqrt(sq/nonzero).real
                else:
                    blstd[i,j,k] = 0.


@jit(nopython=True)
def calcflags_blstd(flags, blstd, blstdmed, blstdstd, sigma, chans, pol):
    """ Flaging based on standard deviation over baseline axis """

    for i in xrange(flags.shape[0]):
        for j in xrange(flags.shape[1]):
            for chan in chans:
                flags[i, j, chan, pol] = blstd[i, chan, pol] < blstdmed + sigma*blstdstd


@jit(nopython=True)
def meddev(arr, md, win=20):
    """ Calculate the median deviation of array """

    for ind in xrange(arr.shape[0]):
        rr = xrange(max(0, ind-win//2), min(len(arr), ind+win//2))
        md[ind] = arr[ind] - np.median(arr[rr])


@jit(nopython=True)
def calcflags_badchslide(flags, meanspec, meddevspec, sigma, chans, pol):
    """ Flagging based on sliding window in channel """

    for i in xrange(flags.shape[0]):
        for j in xrange(flags.shape[1]):
            for chan in chans:
                flags[i, j, chan, pol] = meddevspec[chan-chans[0]] < sigma*meddevspec.std()


@jit(nopython=True)
def calcflags_badtslide(flags, meanlc, meddevlc, sigma, chans, pol):
    """ Flagging based on sliding window in time """

    for i in xrange(data.shape[0]):
        for j in xrange(data.shape[1]):
            for chan in chans:
                flags[i, j, chan, pol] = meddevlc[i] < sigma*meddevlc.std())


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
