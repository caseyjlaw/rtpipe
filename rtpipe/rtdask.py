import dask.array as da
import rtpipe.RT as rt
import rtpipe.parsecal as pc
import rtpipe.parsesdm as ps
import rtlib_cython as rtlib
from functools import partial
import logging
import numpy as np
from numba import jit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)
logger = logging.getLogger('rtpipe')


def dataprep(d, segment):
    data = read_segment(d, segment)

    (u_read, v_read, w_read) = ps.get_uvw_segment(d, segment)

    data = calibrate(data, d, segment)

#    rt.dataflag(d, data)  # can't use cython

    data = meantsub(data) # does not ignore zeros yet

    return data


def calcchunk(length, blocksize):
    """ Calculate chunk tuple given a dimension length and blocksize"""

    ext = np.linspace(0, length, blocksize+1, dtype=int)
    chunks = tuple([(ext[i+1] - ext[i]) for i in range(len(ext)-1) ])
    return chunks


def read_segment(d, segment, blocksize=10):
    """ Read segment of data and chunk it for efficiency of later algorithms """

    data = ps.read_bdf_segment(d, segment)
    chunks0 = calcchunk(data.shape[0], blocksize)
    chunks1 = calcchunk(data.shape[1], blocksize)
    chunks2 = calcchunk(data.shape[2], blocksize)

    chunks = (chunks0, chunks1, chunks2, (1,1))

    return da.from_array(data, chunks=chunks)


def calibrate(data, d, segment):
    sols = pc.telcal_sol(d['gainfile'])   # parse gainfile
    sols.set_selection(d['segmenttimes'][segment].mean(), d['freq']*1e9, rtlib.calc_blarr(d), calname='', pols=d['pols'], radec=(), spwind=[])
    data2 = sols.apply(data)

    return data2


def meantsub(data):
    return data - mean0(data, axis=0)


@jit
def count0(data, axis=0):
    """ Numba-accelerated count of non-zero entries """

    total = 0
    nonzeros = arr2.all(axis=0)
    for ?
        if nonzeros[i,j,k]:
            total += data[i,j,k,axis?]
    

@jit
def mean0(data, axis=0):
    """ Numba-accelerated mean over an axis ignoring zeros """

    mean = data.sum(axis=axis)
    total = count0(data, axis=axis)

    return mean/total


@jit
def std0(data, axis=0):
    """ Numba-accelerated mean over an axis ignoring zeros """

    # or make a dask graph?

    std = data**2.sum(axis=axis)
    total = count0(data, axis=axis)

    return (std/total)**(1/2.)


def search():
    dedisppart = partial(correct_dm, d, dm)   # moves in fresh data
    dedispresults = resamppool.map(dedisppart, blranges)

    dtlast = 1
    for dtind in xrange(len(d['dtarr'])):
        dt = d['dtarr'][dtind]
                    
        if dt > 1:
            resample = dt/dtlast
            resamppart = partial(correct_dt, d, resample)   # corrects in place
            resampresults = resamppool.map(resamppart, blranges)
            dtlast = dt

        nskip_dm = ((d['datadelay'][-1] - d['datadelay'][dmind]) / dt) * (d['segment'] != 0)  # nskip=0 for first segment
        searchints = (d['readints'] - d['datadelay'][dmind]) / dt - nskip_dm
        image1part = partial(image1, d, u, v, w, dmind, dtind, beamnum)
        nchunkdt = min(searchints, max(d['nthread'], d['nchunk']/dt))  # parallelize in range bounded by (searchints, nthread)
        irange = [(nskip_dm + searchints*chunk/nchunkdt, nskip_dm + searchints*(chunk+1)/nchunkdt) for chunk in range(nchunkdt)]
        imageresults = resamppool.map(image1part, irange)

        for imageresult in imageresults:
            for kk in imageresult.keys():
                cands[kk] = imageresult[kk]
