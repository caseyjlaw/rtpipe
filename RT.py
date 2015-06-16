import rtpipe.parsems as pm
import rtpipe.parsecal as pc
import rtpipe.parsesdm as ps
import rtlib_cython as rtlib
import multiprocessing as mp
import multiprocessing.sharedctypes as mps
from contextlib import closing
import numpy as n
from scipy.special import erf
import casautil, os, pickle, glob, time
import logging

# setup CASA and logging
qa = casautil.tools.quanta()
logger = logging.getLogger('rtpipe')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

def pipeline(d, segment, reproducecand=()):
    """ Transient search pipeline running on single node.
    Processes a single segment of data (where a single bgsub, (u,v,w), etc. can be used).
    Searches completely, independently, and saves candidates.
    reproducecand is tuple of (dmind, dtind) or (candint, dmind, dtind). Former returns corrected data, latter images and phases data.

    Stages:
    0) Take dictionary that defines metadata and search params
    -- This defines state of pipeline, including times, uv extent, pipeline search parameters, etc.
    1) Read data
    -- Overlapping reads needed to maintain sensitivity to all DMs at all times
    2) Prepare data
    -- Reads/applies telcal/CASA solutions, flags, bg time subtraction
    3) Search using all threads
    -- Option for plug-and-play detection algorithm and multiple filters
    4) Save candidate and noise info, if requested
    """

    ####    ####    ####    ####
    # 1) Read data
    ####    ####    ####    ####

    os.chdir(d['workdir'])

    candsfile = 'cands_' + d['fileroot'] + '_sc' + str(d['scan']) + 'seg' + str(segment) + '.pkl'
    if d['savecands'] and os.path.exists(candsfile) and len(reproducecand) == 0:
        logger.error('candsfile %s already exists. Ending processing...' % candsfile)
        return 0
    
    if d['dataformat'] == 'ms':   # CASA-based read
        segread = pm.readsegment(d, segment)
        readints = len(segread[0])
#        data = n.empty( (readints, d['nbl'], d['nchan'], d['npol']), dtype='complex64', order='C')  # orig
        data_mem = mps.RawArray(mps.ctypes.c_float, long(readints*d['nbl']*d['nchan']*d['npol'])*2)  # 'long' type needed to hold whole (2 min, 5 ms) scans
        data = numpyview(data_mem, 'complex64', (readints, d['nbl'], d['nchan'], d['npol']))
        data[:] = segread[0]
        (u, v, w) = (segread[1][readints/2], segread[2][readints/2], segread[3][readints/2])  # mid int good enough for segment. could extend this to save per chunk
        del segread

    elif d['dataformat'] == 'sdm':
        t0 = d['segmenttimes'][segment][0]
        t1 = d['segmenttimes'][segment][1]
        readints = n.round(24*3600*(t1 - t0)/d['inttime'], 0).astype(int)/d['read_tdownsample']
        assert readints > 0

        # for shared mem
        data_mem = mps.RawArray(mps.ctypes.c_float, long(readints*d['nbl']*d['nchan']*d['npol'])*2)  # 'long' type needed to hold whole (2 min, 5 ms) scans
        data = numpyview(data_mem, 'complex64', (readints, d['nbl'], d['nchan'], d['npol']))

        data[:] = ps.read_bdf_segment(d, segment)
        (u,v,w) = ps.get_uvw_segment(d, segment)
    else:
        logger.error('Data format %s not supported.' % d['dataformat'])

    ####    ####    ####    ####
    # 2) Prepare data
    ####    ####    ####    ####

    # calibrate data
    try:
        sols = pc.casa_sol(d['gainfile'], flagants=d['flagantsol'])
        sols.parsebp(d['bpfile'])
        sols.setselection(d['segmenttimes'][segment].mean(), d['freq']*1e9, radec=d['radec'])
        sols.apply(data, d['blarr'])
    except IOError:
        logger.info('Calibration file not found. Proceeding with no calibration applied.')

    # flag data
    if d['flagmode'] == 'standard':
        dataflagpool(data_mem, d)
    else:
        logger.info('No real-time flagging.')

    # mean t vis subtration
    if d['timesub'] == 'mean':
        meantsubpool(data_mem, d)
    else:
        logger.info('No mean time subtraction.')

    if d['savenoise']:
        noisepickle(d, data, u, v, w)      # save noise pickle

    # optionally phase to new location
    if any([d['l0'], d['m0']]):
        logger.info('Rephasing data to (l, m)=(%.3f, %.3f).' % (d['l0'], d['m0']))
        rtlib.phaseshift_threaded(data, d, d['l0'], d['m0'], u, v)

    ####    ####    ####    ####
    # 3) Search using all threads
    ####    ####    ####    ####
    if len(reproducecand) == 0:
        logger.info('Starting search...')
        cands = search(d, data, u, v, w)

        ####    ####    ####    ####
        # 4) Save candidate info
        ####    ####    ####    ####
        if d['savecands']:
            logger.info('Saving %d candidates...' % (len(cands)))
            savecands(d, cands)

        return len(cands)

    elif len(reproducecand) == 2:  # reproduce data
        logger.info('Reproducing data...')
        dmind, dtind = reproducecand
        d['dmarr'] = [d['dmarr'][dmind]]
        d['dtarr'] = [d['dtarr'][dtind]]
        data = reproduce(d, data_mem, u, v, w)
        return data

    elif len(reproducecand) == 3:  # reproduce candidate image and data
        logger.info('Reproducing candidate...')
        reproduceint, dmind, dtind = reproducecand
        d['dmarr'] = [d['dmarr'][dmind]]
        d['dtarr'] = [d['dtarr'][dtind]]
        im, data = reproduce(d, data_mem, u, v, w, reproduceint)
        return im, data

    else:
        logger.error('reproducecand not in expected format: %s' % reproducecand)

def meantsubpool(data_mem, d):
    """ Parallelized mean t visibility subtraction.
    """

    logger.info('Subtracting mean visibility in time...')
    blranges = [(d['nbl'] * t/d['nthread'], d['nbl']*(t+1)/d['nthread']) for t in range(d['nthread'])]
    with closing(mp.Pool(d['nthread'], initializer=initpool2, initargs=(data_mem,))) as pool:
        for blr in blranges:
            pool.apply_async(meantsub, [blr, d])

def meantsub(blr, d):
    """ Wrapper function for rtlib.meantsub
    Assumes data_mem is global mps.RawArray
    """
    readints = len(data_mem)/(d['nbl']*d['nchan']*d['npol']*2)
    data = numpyview(data_mem, 'complex64', (readints, d['nbl'], d['nchan'], d['npol']))

    rtlib.meantsub(data, blr)

def dataflagpool(data_mem, d):
    """ Parallelized flagging
    """

    logger.info('Flagging data...')
    chperspw = len(d['freq_orig'])/len(d['spw'])
    with closing(mp.Pool(d['nthread'], initializer=initpool2, initargs=(data_mem,))) as pool:
        resultd = {}
        for pol in range(d['npol']):
            for spw in range(d['nspw']):
                freqs = d['freq_orig'][spw*chperspw:(spw+1)*chperspw]  # find chans for spw. only works for 2 or more sb
                chans = n.array([i for i in xrange(len(d['freq'])) if d['freq'][i] in freqs])
                resultd[(spw,pol)] = pool.apply_async(dataflag, [chans, pol, d, 20., 'badcht', 0.3])
#                resultd[(spw,pol)] = pool.apply_async(dataflag, [chans, pol, d, 10., 'badcht', 0.05])
        for kk in resultd.keys():
            result = resultd[kk].get()
            logger.info(result)

        resultd = {}
        for spw in range(d['nspw']):
            freqs = d['freq_orig'][spw*chperspw:(spw+1)*chperspw]  # find chans for spw. only works for 2 or more sb
            chans = n.array([i for i in xrange(len(d['freq'])) if d['freq'][i] in freqs])
            resultd[spw] = pool.apply_async(dataflag, [chans, 0, d, 3., 'badap', 0.2]) # pol not used here
        for kk in resultd.keys():
            result = resultd[kk].get()
            logger.info(result)

        resultd = {}
        for pol in range(d['npol']):
            for spw in range(d['nspw']):
                freqs = d['freq_orig'][spw*chperspw:(spw+1)*chperspw]  # find chans for spw. only works for 2 or more sb
                chans = n.array([i for i in xrange(len(d['freq'])) if d['freq'][i] in freqs])
#                resultd[(spw,pol)] = pool.apply_async(dataflag, [chans, pol, d, 10.0, 'blstd', 0.2])
                resultd[(spw,pol)] = pool.apply_async(dataflag, [chans, pol, d, 3.0, 'blstd', 0.05])
        for kk in resultd.keys():
            result = resultd[kk].get()
            logger.info(result)

        resultd = {}
        for pol in range(d['npol']):
            for spw in range(d['nspw']):
                freqs = d['freq_orig'][spw*chperspw:(spw+1)*chperspw]  # find chans for spw. only works for 2 or more sb
                chans = n.array([i for i in xrange(len(d['freq'])) if d['freq'][i] in freqs])
                resultd[(spw,pol)] = pool.apply_async(dataflag, [chans, pol, d, 15., 'badcht', 0.3])
#                resultd[(spw,pol)] = pool.apply_async(dataflag, [chans, pol, d, 7., 'badcht', 0.1])
        for kk in resultd.keys():
            result = resultd[kk].get()
            logger.info(result)

            # resultd = {}
            # for pol in range(d['npol']):
            #     for spw in range(d['nspw']):
            #         freqs = d['freq_orig'][spw*chperspw:(spw+1)*chperspw]  # find chans for spw. only works for 2 or more sb
            #         chans = n.array([i for i in xrange(len(d['freq'])) if d['freq'][i] in freqs])
            #         resultd[(spw,pol)] = pool.apply_async(dataflag, [chans, pol, d, 4., 'ring', 0.2])
            # for kk in resultd.keys():
            #     result = resultd[kk].get()
#                logger.debug(kk, result, result/(0.5*data.size))

def dataflag(chans, pol, d, sig, mode, conv):
    """ Wrapper function to get shared memory as numpy array into pool
    Assumes data_mem is global mps.RawArray
    """
    readints = len(data_mem)/(d['nbl']*d['nchan']*d['npol']*2)
    data = numpyview(data_mem, 'complex64', (readints, d['nbl'], d['nchan'], d['npol']))
#    data = n.ma.masked_array(data, data==0j)  # this causes massive overflagging on 14sep03 data

    return rtlib.dataflag(data, chans, pol, d, sig, mode, conv)

def search(d, data, u, v, w):
    """ Search function.
    Queues all trials with multiprocessing.
    Assumes shared memory system with single uvw grid for all images.
    """

    beamnum = 0
    cands = {}
    resultlist = []

    candsfile = 'cands_' + d['fileroot'] + '_sc' + str(d['scan']) + 'seg' + str(d['segment']) + '.pkl'
    if d['savecands'] and os.path.exists(candsfile):
        logger.warn('candsfile %s already exists' % candsfile)
        return cands

    readints = len(data)
#    data_resamp_mem = [mps.RawArray(mps.ctypes.c_float, (readints*d['nbl']*d['nchan']*d['npol'])*2) for resamp in range(len(d['dtarr']))]    # simple shared array
#    data_resamp = [numpyview(data_resamp_mem[resamp], 'complex64', (readints, d['nbl'], d['nchan'], d['npol'])) for resamp in range(len(d['dtarr']))]
#    data_resamp[0][:] = data[:]
#    del data

#    readints = len(data_mem)/(d['nbl']*d['nchan']*d['npol']*2)
#    data = numpyview(data_mem, 'complex64', (readints, d['nbl'], d['nchan'], d['npol']))

    data_resamp_mem = mps.RawArray(mps.ctypes.c_float, (readints*d['nbl']*d['nchan']*d['npol'])*2)
    data_resamp = numpyview(data_resamp_mem, 'complex64', (readints, d['nbl'], d['nchan'], d['npol']))

    # make wterm kernels
    if d['searchtype'] == 'image2w':
        wres = 100
        npix = max(d['npixx_full'], d['npixy_full'])
        bls, uvkers = rtlib.genuvkernels(w, wres, npix, d['uvres'], thresh=0.05)

    # SUBMITTING THE LOOPS
    if n.any(data):
        logger.info('Searching in %d chunks with %d threads' % (d['nchunk'], d['nthread']))
        logger.info('Dedispering to max (DM, dt) of (%d, %d) ...' % (d['dmarr'][-1], d['dtarr'][-1]) )
        with closing(mp.Pool(d['nthread'], initializer=initpool, initargs=(data_resamp_mem,))) as pool:
            for dmind in xrange(len(d['dmarr'])):
                for dtind in xrange(len(d['dtarr'])):
                    logger.info('Starting (%d,%d)' % (d['dmarr'][dmind], d['dtarr'][dtind]),)
                    data_resamp[:] = data[:]
                    blranges = [(d['nbl'] * t/d['nthread'], d['nbl']*(t+1)/d['nthread']) for t in range(d['nthread'])]
                    for blr in blranges:
                        result = pool.apply_async(correct_dmdt, [d, dmind, dtind, blr])
                        resultlist.append(result)
                    for result in resultlist:
                        result.wait()
                    resultlist = []

                    # set dm-dependent starting int for segment
                    nskip_dm = (d['datadelay'][-1] - d['datadelay'][dmind]) * (d['segment'] != 0)  # nskip=0 for first segment
                    searchints = readints - d['datadelay'][dmind] - nskip_dm

                    # submit jobs in chunks of time
                    for chunk in range(d['nchunk']):
                        i0 = nskip_dm + (searchints/d['dtarr'][dtind])*chunk/d['nchunk']
                        i1 = nskip_dm + (searchints/d['dtarr'][dtind])*(chunk+1)/d['nchunk']

                        if d['searchtype'] == 'image1':
                            result = pool.apply_async(image1, [d, i0, i1, u, v, w, dmind, dtind, beamnum])
                            resultlist.append(result)
                        elif d['searchtype'] == 'image2':
                            result = pool.apply_async(image2, [d, i0, i1, u, v, w, dmind, dtind, beamnum])
                            resultlist.append(result)
                        elif d['searchtype'] == 'image2w':
                            result = pool.apply_async(image2w, [d, i0, i1, u, v, w, dmind, dtind, beamnum, bls, uvkers])
                            resultlist.append(result)

                    # COLLECTING THE RESULTS per DM loop. Clears the way for overwriting data_resamp
                    for result in resultlist:
                        feat = result.get()
                        for kk in feat.keys():
                            cands[kk] = feat[kk]
#                pool.join()
    else:
        logger.warn('Data for processing is zeros. Moving on...')

    logger.info('Found %d cands in scan %d segment %d of %s. ' % (len(cands), d['scan'], d['segment'], d['filename']))
    return cands

def reproduce(d, data_resamp_mem, u, v, w, candint=-1, twindow=30):
    """ Reproduce function, much like search.
    If no candint is given, it returns resampled data. Otherwise, returns image and rephased data.
    """

    dmind = 0; dtind = 0
    readints = len(data_resamp_mem)/(d['nbl']*d['nchan']*d['npol']*2)
    data_resamp = numpyview(data_resamp_mem, 'complex64', (readints, d['nbl'], d['nchan'], d['npol']))

    with closing(mp.Pool(1, initializer=initpool, initargs=(data_resamp_mem,))) as pool:
        # dedisperse
        logger.info('Dedispersing with DM=%.1f, dt=%d...' % (d['dmarr'][dmind], d['dtarr'][dtind]))
        pool.apply(correct_dmdt, [d, dmind, dtind, (0,d['nbl'])])

        # set up image
        if d['searchtype'] == 'image1':
            npixx = d['npixx']
            npixy = d['npixy']
        elif d['searchtype'] == 'image2':
            npixx = d['npixx_full']
            npixy = d['npixy_full']

        if candint > -1:
            # image
            logger.info('Imaging int %d with %d %d pixels...' % (candint, npixx, npixy))
            im = pool.apply(image1wrap, [d, u, v, w, npixx, npixy, candint/d['dtarr'][dtind]])

            snrmin = im.min()/im.std()
            snrmax = im.max()/im.std()
            logger.info('Made image with SNR min, max: %.1f, %.1f' % (snrmin, snrmax))
            if snrmax > -1*snrmin:
                peakl, peakm = n.where(im == im.max())
            else:
                peakl, peakm = n.where(im == im.min())
            l1 = (npixx/2. - peakl[0])/(npixx*d['uvres'])
            m1 = (npixy/2. - peakm[0])/(npixy*d['uvres'])

            # rephase and trim interesting ints out
            logger.info('Rephasing to peak...')
            pool.apply(move_phasecenter, [d, l1, m1, u, v])
            minint = max(candint/d['dtarr'][dtind]-twindow/2, 0)
            maxint = min(candint/d['dtarr'][dtind]+twindow/2, len(data_resamp)/d['dtarr'][dtind])

            return(im, data_resamp[minint:maxint].mean(axis=1))
        else:
            return data_resamp

def lightcurve(d, l1, m1):
    """ Makes lightcurve at given (l1, m1)
    """

    ####    ####    ####    ####
    # 1) Read data
    ####    ####    ####    ####

    os.chdir(d['workdir'])

#    for segment in range(d['nsegments']):
    for segment in [0,1]:
        t0 = d['segmenttimes'][segment][0]
        t1 = d['segmenttimes'][segment][1]
        readints = n.round(24*3600*(t1 - t0)/d['inttime'], 0).astype(int)/d['read_tdownsample']
        data_mem = mps.RawArray(mps.ctypes.c_float, long(readints*d['nbl']*d['nchan']*d['npol'])*2)  # 'long' type needed to hold whole (2 min, 5 ms) scans
        data = numpyview(data_mem, 'complex64', (readints, d['nbl'], d['nchan'], d['npol']))

        data[:] = ps.read_bdf_segment(d, segment)
        (u,v,w) = ps.get_uvw_segment(d, segment)

    ####    ####    ####    ####
    # 2) Prepare data
    ####    ####    ####    ####

        # calibrate data
        try:
            sols = pc.casa_sol(d['gainfile'], flagants=d['flagantsol'])
            sols.parsebp(d['bpfile'])
            sols.setselection(d['segmenttimes'][segment].mean(), d['freq']*1e9, radec=d['radec'])
            sols.apply(data, d['blarr'])
        except IOError:
            logger.error('Calibration file not found. Proceeding with no calibration applied.')

        # flag data
        if d['flagmode'] == 'standard':
            dataflagpool(data_mem, d)
        else:
            logger.info('No real-time flagging.')

        # mean t vis subtration
        if d['timesub'] == 'mean':
            meantsubpool(data_mem, d)
        else:
            logger.info('No mean time subtraction.')

        # no dedispersion yet

        # only returns significant images for now
        vals = rtlib.imgallfullfilterxyflux(n.outer(u, d['freq']/d['freq_orig'][0]), n.outer(v, d['freq']/d['freq_orig'][0]), data, d['npixx'], d['npixy'], d['uvres'], d['sigma_image1'])

        rtlib.phaseshift_threaded(data, d, l1, m1, u, v)

        if segment == 0:
            phaseddata = n.ma.masked_array(data, data==0j).mean(axis=1)
            images = vals[0]
        else:
            phaseddata = n.concatenate( (phaseddata, n.ma.masked_array(data, data==0j).mean(axis=1)), axis=0)
            images = images+vals[0]

    return images, phaseddata
    
def set_pipeline(filename, scan, fileroot='', paramfile='', **kwargs):
    """ Function defines pipeline state for search. Takes data/scan as input.
    fileroot is base name for associated products (cal files, noise, cands). if blank, it is set to filename.
    paramfile is name of file that defines all pipeline parameters (python-like syntax).
    kwargs used to overload paramfile definitions.
    Many parameters take 0 as default, which auto-defines ideal parameters. 
    This definition does not yet consider memory/cpu/time limitations.
    nsegments defines how to break jobs in time. nchunk defines how many jobs are sent to nthreads.
    """

    
    workdir = os.path.split(os.path.abspath(filename))[0]

    # option of not writing log file (need to improve later)
    if 'nologfile' in kwargs.keys(): 
        pass
    else:
        fh = logging.FileHandler(os.path.join(workdir, 'rtpipe_%d.log' % int(round(time.time()))))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

    # define metadata (state) dict. chans/spw is special because it goes in to get_metadata call
    if 'chans' in kwargs.keys(): 
        chans=kwargs['chans']
    else:
        chans = []
    if 'spw' in kwargs.keys(): 
        spw=kwargs['spw']
    else:
        spw = []
    if 'read_fdownsample' in kwargs.keys(): 
        rfd=kwargs['read_fdownsample']
    else:
        rfd = 1
    if 'datacol' in kwargs.keys(): 
        datacol=kwargs['datacol']
    else:
        datacol = []

    # then get all metadata
    if os.path.exists(os.path.join(filename, 'Antenna.xml')):
        d = ps.get_metadata(filename, scan, chans=chans, spw=spw, read_fdownsample=rfd, params=paramfile)   # can take file name or Params instance
        d['dataformat'] = 'sdm'
    else:
        d = pm.get_metadata(filename, scan, chans=chans, spw=spw, read_fdownsample=rfd, params=paramfile)
        d['dataformat'] = 'ms'

    # overload with provided kwargs
    for key in kwargs.keys():
        if kwargs.keys().index(key) == 0: logger.info('')
        logger.info('Setting %s to %s' % (key, kwargs[key]))
        d[key] = kwargs[key]

    # define rootname for in/out cal/products
    if fileroot:
        d['fileroot'] = fileroot
    else:
        d['fileroot'] = os.path.split(os.path.abspath(filename))[1]

    # autodetect calibration products
    if not d['gainfile']:
        filelist = glob.glob(os.path.join(d['workdir'], d['fileroot'] + '.g?'))
        if len(filelist):
            filelist.sort()
            d['gainfile'] = filelist[-1]
            logger.info('Autodetecting cal files... gainfile set to %s.' % d['gainfile'])
    if not d['bpfile']:
        filelist = glob.glob(os.path.join(d['workdir'], d['fileroot'] + '.b?'))
        if len(filelist):
            filelist.sort()
            d['bpfile'] = filelist[-1]
            logger.info('Autodetecting cal files... bpfile set to %s.' % d['bpfile'])

    # supported features: snr1, immax1, l1, m1
    if d['searchtype'] == 'image1':
        d['features'] = ['snr1', 'immax1', 'l1', 'm1']   # features returned by image1
    elif 'image2' in d['searchtype']:
        d['features'] = ['snr1', 'immax1', 'l1', 'm1', 'snr2', 'immax2', 'l2', 'm2']   # features returned by image1
    d['featureind'] = ['segment', 'int', 'dmind', 'dtind', 'beamnum']  # feature index. should be stable.

    # redefine good antennas
    if len(d['excludeants']):
        for ant in d['excludeants']:
            d['ants'].remove(ant)
        d['nants'] = len(n.unique(d['blarr']))
        d['blarr'] = n.array( [(ant1,ant2) for (ant1,ant2) in d['blarr'] if ((ant1 not in d['excludeants']) and (ant2 not in d['excludeants']))] )
        d['nbl'] = len(d['blarr'])

    # set imaging parameters to use
    if d['uvres'] == 0:
        d['uvres'] = d['uvres_full']
    else:
        urange = d['urange'][scan]*(d['freq'].max()/d['freq_orig'][0])   # uvw from get_uvw already in lambda at ch0
        vrange = d['vrange'][scan]*(d['freq'].max()/d['freq_orig'][0])
        powers = n.fromfunction(lambda i,j: 2**i*3**j, (14,10), dtype='int')   # power array for 2**i * 3**j
        rangex = n.round(d['uvoversample']*urange).astype('int')
        rangey = n.round(d['uvoversample']*vrange).astype('int')
        largerx = n.where(powers-rangex/d['uvres'] > 0, powers, powers[-1,-1])
        p2x, p3x = n.where(largerx == largerx.min())
        largery = n.where(powers-rangey/d['uvres'] > 0, powers, powers[-1,-1])
        p2y, p3y = n.where(largery == largery.min())
        d['npixx_full'] = (2**p2x * 3**p3x)[0]
        d['npixy_full'] = (2**p2y * 3**p3y)[0]

    if d['npix'] == 0:
        if d.has_key('npix_max'):   # optional 'do not exceed' image specification
            d['npixx'] = min(d['npix_max'], d['npixx_full'])
            d['npixy'] = min(d['npix_max'], d['npixy_full'])
        else:    # otherwise, go with full res
            d['npixx'] = d['npixx_full']
            d['npixy'] = d['npixy_full']
        d['npix'] = max(d['npixx'], d['npixy'])   # this used to define fringe time
    else:
        d['npixx'] = d['npix']
        d['npixy'] = d['npix']

    # define dmarr, if not already
    if len(d['dmarr']) == 0:
        if d.has_key('dm_maxloss') and d.has_key('maxdm') and d.has_key('dm_pulsewidth'):
            d['dmarr'] = calc_dmgrid(d, maxloss=d['dm_maxloss'], maxdm=d['maxdm'], dt=d['dm_pulsewidth'])
            logger.info('Calculated %d dms for max sensitivity loss %.2f, maxdm %d pc/cm3, and pulse width %d ms' % (len(d['dmarr']), d['dm_maxloss'], d['maxdm'], d['dm_pulsewidth']/1000))
        else:
            d['dmarr'] = [0]
            logger.info('Can\'t calculate dm grid without dm_maxloss, maxdm, and dm_pulsewidth defined. Setting to [0].')

    # define times for data to read
    d['t_overlap'] = rtlib.calc_delay(d['freq'], d['inttime'], max(d['dmarr'])).max()*d['inttime']   # time of overlap for total dm coverage at segment boundaries
    d['datadelay'] = [rtlib.calc_delay(d['freq'], d['inttime'],dm).max() for dm in d['dmarr']]
    d['nints'] = d['nints'] - d['nskip']

    if d['nsegments'] == 0:
        fringetime = calc_fringetime(d)
        d['nsegments'] = max(1, d['scale_nsegments']*int(d['inttime']*(d['nints']-d['nskip'])/(fringetime-d['t_overlap'])))
#        stopdts = n.arange(d['nskip']*d['inttime']+d['t_overlap'], d['nints']*d['inttime'], fringetime-d['t_overlap'])[1:] # old way
#        startdts = n.concatenate( ([d['nskip']*d['inttime']], stopdts[:-1]-d['t_overlap']) )

    stopdts = n.linspace(d['nskip']*d['inttime']+d['t_overlap'], d['nints']*d['inttime'], d['nsegments']+1)[1:]
    startdts = n.concatenate( ([d['nskip']*d['inttime']], stopdts[:-1]-d['t_overlap']) )

    segmenttimes = []
    for (startdt, stopdt) in zip(startdts, stopdts):
        starttime = qa.getvalue(qa.convert(qa.time(qa.quantity(d['starttime_mjd']+startdt/(24.*60*60),'d'),form=['ymd'], prec=9)[0], 's'))[0]/(24*3600)
        stoptime = qa.getvalue(qa.convert(qa.time(qa.quantity(d['starttime_mjd']+stopdt/(24.*60*60), 'd'), form=['ymd'], prec=9)[0], 's'))[0]/(24*3600)
        segmenttimes.append((starttime, stoptime))
    d['segmenttimes'] = n.array(segmenttimes)
    d['t_segment'] = 24*3600*(d['segmenttimes'][0,1]-d['segmenttimes'][0,0])

    # scaling of number of integrations beyond dt=1
    assert all(d['dtarr'])
    dtfactor = n.sum([1./i for i in d['dtarr']])    # assumes dedisperse-all algorithm

    # calculate number of thermal noise candidates per segment
    ntrials = d['t_segment']/(d['inttime']*d['read_tdownsample']) * dtfactor * len(d['dmarr']) * d['npixx'] * d['npixy']
    qfrac = 1 - (erf(d['sigma_image1']/n.sqrt(2)) + 1)/2.
    nfalse = int(qfrac*ntrials)

    # split imaging into chunks. ideally one per thread, but can modify to fit available memory
    if d['nchunk'] == 0:
        d['nchunk'] = d['nthread']

    logger.info('')
    logger.info('Pipeline summary:')
    logger.info('\t Products saved with %s. Calibration files set to (%s, %s)' % (d['fileroot'], d['gainfile'], d['bpfile']))
    logger.info('\t Using %d segment%s of %d ints (%.1f s) with overlap of %.1f s' % (d['nsegments'], "s"[not d['nsegments']-1:], d['t_segment']/(d['inttime']*d['read_tdownsample']), d['t_segment'], d['t_overlap']))
    if d['t_overlap'] > d['t_segment']/3.:
        logger.info('\t\t Lots of segments needed, since Max DM sweep (%.1f s) close to segment size (%.1f s)' % (d['t_overlap'], d['t_segment']))
    logger.info('\t Downsampling in time/freq by %d/%d and skipping %d ints from start of scan.' % (d['read_tdownsample'], d['read_fdownsample'], d['nskip']))
    logger.info('\t Excluding ants %s' % (d['excludeants']))
    logger.info('')

    logger.info('\t Search with %s and threshold %.1f.' % (d['searchtype'], d['sigma_image1']))
    logger.info('\t Using %d DMs from %.1f to %.1f and dts %s.' % (len(d['dmarr']), min(d['dmarr']), max(d['dmarr']), d['dtarr']))
    logger.info('\t Using uvgrid npix=(%d,%d) and res=%d.' % (d['npixx'], d['npixy'], d['uvres']))
    logger.info('\t Expect %d thermal false positives per segment.' % nfalse)

    (vismem, immem) = calc_memory_footprint(d)
    if d.has_key('memory_limit'):    # if preference given, then test
        if vismem+immem > d['memory_limit']:
            logger.info('')
            while vismem+immem > d['memory_limit']:
                logger.info('Doubling nchunk from %d to fit in %d GB memory limit.' % (d['nchunk'], d['memory_limit']))
                d['nchunk'] = 2*d['nchunk']
                (vismem, immem) = calc_memory_footprint(d)

    logger.info('')
    logger.info('\t Visibility memory usage is %.1f GB/segment' % vismem)
    logger.info('\t Imaging in %d chunk%s using max of %.1f GB/segment' % (d['nchunk'], "s"[not d['nsegments']-1:], immem))
    logger.info('\t Grand total memory usage: %d GB/segment' % (vismem + immem))

    return d

def calc_memory_footprint(d, headroom=2.):
    """ Given pipeline state dict, this function calculates the memory required
    to store visibilities and make images.
    headroom scales memory size from ideal to realistic. only used for vismem.
    Returns tuple of (vismem, immem) in units of GB.
    """

    toGB = 8/1024.**3   # number of complex64s to GB
    dtfactor = n.sum([1./i for i in d['dtarr']])    # assumes dedisperse-all algorithm
    nints = d['t_segment']/(d['inttime']*d['read_tdownsample'])

    vismem = headroom * dtfactor * (nints * d['nbl'] * d['nchan'] * d['npol']) * toGB
    immem = d['nthread'] * (nints/d['nchunk'] * d['npixx'] * d['npixy']) * toGB
    return (vismem, immem)

def calc_fringetime(d):
    """ Estimate largest time span of a "segment".
    A segment is the maximal time span that can be have a single bg fringe subtracted and uv grid definition.
    Max fringe window estimated for 5% amp loss at first null averaged over all baselines. Assumes dec=+90, which is conservative.
    Returns time in seconds that defines good window.
    """

    maxbl = d['uvres']*d['npix']/2    # fringe time for imaged data only
    fringetime = 0.5*(24*3600)/(2*n.pi*maxbl/25.)   # max fringe window in seconds
    return fringetime

def correct_dmdt(d, dmind, dtind, blr):
    """ Dedisperses and resamples data *in place*.
    Drops edges, since it assumes that data is read with overlapping chunks in time.
    """

    readints = len(data_resamp_mem)/(d['nbl']*d['nchan']*d['npol']*2)
    data_resamp = numpyview(data_resamp_mem, 'complex64', (readints, d['nbl'], d['nchan'], d['npol']))

    rtlib.dedisperse_resample(data_resamp, d['freq'], d['inttime'], d['dmarr'][dmind], d['dtarr'][dtind], blr, verbose=0)        # dedisperses data.

def calc_lm(d, im, pix=()):
    """ Helper function to calculate location of image pixel in (l,m) coords.
    Assumes peak pixel, but input can be provided in pixel units.
    """

    if len(pix) == 0:
        peakl, peakm = n.where(im == im.max())
        peakl = peakl[0]; peakm = peakm[0]
    elif len(pix) == 2:
        peakl, peakm = pix
    npixx, npixy = im.shape
    l1 = (npixx/2. - peakl)/(npixx*d['uvres'])
    m1 = (npixy/2. - peakm)/(npixy*d['uvres'])
    return l1, m1

def move_phasecenter(d, l1, m1, u, v):
    """ Handler function for phaseshift_threaded
    """

    readints = len(data_resamp_mem)/(d['nbl']*d['nchan']*d['npol']*2)
    data_resamp = numpyview(data_resamp_mem, 'complex64', (readints, d['nbl'], d['nchan'], d['npol']))

    rtlib.phaseshift_threaded(data_resamp, d, l1, m1, u, v)

def calc_dmgrid(d, maxloss=0.05, dt=3000., mindm=0., maxdm=2000.):
    """ Function to calculate the DM values for a given maximum sensitivity loss.
    maxloss is sensitivity loss tolerated by dm bin width. dt is assumed pulse width in microsec.
    """

    # parameters
    tsamp = d['inttime']*1e6  # in microsec
    k = 8.3
    freq = d['freq'].mean()  # central (mean) frequency in GHz
    bw = 1e3*(d['freq'][-1] - d['freq'][0])
    ch = 1e3*(d['freq'][1] - d['freq'][0])  # channel width in MHz

    # width functions and loss factor
    dt0 = lambda dm: n.sqrt(dt**2 + tsamp**2 + ((k*dm*ch)/(freq**3))**2)
    dt1 = lambda dm, ddm: n.sqrt(dt**2 + tsamp**2 + ((k*dm*ch)/(freq**3))**2 + ((k*ddm*bw)/(freq**3.))**2)
    loss = lambda dm, ddm: 1-n.sqrt(dt0(dm)/dt1(dm,ddm))

    # iterate over dmgrid to find optimal dm values. go higher than maxdm to be sure final list includes full range.
    dmgrid = n.arange(mindm, maxdm, 0.05)
    dmgrid_final = [dmgrid[0]]
    for i in range(len(dmgrid)):
        ddm = (dmgrid[i] - dmgrid_final[-1])/2.
        ll = loss(dmgrid[i],ddm)
        if ll > maxloss:
            dmgrid_final.append(dmgrid[i])

    return dmgrid_final

def image1(d, i0, i1, u, v, w, dmind, dtind, beamnum):
    """ Parallelizable function for imaging a chunk of data for a single dm.
    Assumes data is dedispersed and resampled, so this just images each integration.
    Simple one-stage imaging that returns dict of params.
    returns dictionary with keys of cand location and values as tuple of features
    """

    readints = len(data_resamp_mem)/(d['nbl']*d['nchan']*d['npol']*2)
    data_resamp = numpyview(data_resamp_mem, 'complex64', (readints, d['nbl'], d['nchan'], d['npol']))

    ims,snr,candints = rtlib.imgallfullfilterxyflux(n.outer(u, d['freq']/d['freq_orig'][0]), n.outer(v, d['freq']/d['freq_orig'][0]), data_resamp[i0:i1], d['npixx'], d['npixy'], d['uvres'], d['sigma_image1'])

    feat = {}
    for i in xrange(len(candints)):
        if snr[i] > 0:
            peakl, peakm = n.where(ims[i] == ims[i].max())
        else:
            peakl, peakm = n.where(ims[i] == ims[i].min())
        l1 = (d['npixx']/2. - peakl[0])/(d['npixx']*d['uvres'])
        m1 = (d['npixy']/2. - peakm[0])/(d['npixy']*d['uvres'])
        logger.info('Got one!  Int=%d, DM=%d, dt=%d: SNR_im=%.1f @ (%.2e,%.2e).' % ((i0+candints[i])*d['dtarr'][dtind], d['dmarr'][dmind], d['dtarr'][dtind], snr[i], l1, m1))
        candid =  (d['segment'], (i0+candints[i])*d['dtarr'][dtind], dmind, dtind, beamnum)

        # assemble feature in requested order
        ff = []
        for feature in d['features']:
            if feature == 'snr1':
                ff.append(snr[i])
            elif feature == 'immax1':
                if snr[i] > 0:
                    ff.append(ims[i].max())
                else:
                    ff.append(ims[i].min())
            elif feature == 'l1':
                ff.append(l1)
            elif feature == 'm1':
                ff.append(m1)

        feat[candid] = list(ff)
    return feat

def image2(d, i0, i1, u, v, w, dmind, dtind, beamnum):
    """ Parallelizable function for imaging a chunk of data for a single dm.
    Assumes data is dedispersed and resampled, so this just images each integration.
    Two-stage imaging uses ideal uv coverage in second image.
    returns dictionary with keys of cand location and values as tuple of features
    """

    readints = len(data_resamp_mem)/(d['nbl']*d['nchan']*d['npol']*2)
    data_resamp = numpyview(data_resamp_mem, 'complex64', (readints, d['nbl'], d['nchan'], d['npol']))

    ims,snr,candints = rtlib.imgallfullfilterxy(n.outer(u, d['freq']/d['freq_orig'][0]), n.outer(v, d['freq']/d['freq_orig'][0]), data_resamp[i0:i1], d['npixx'], d['npixy'], d['uvres'], d['sigma_image1'])

    feat = {}
    for i in xrange(len(candints)):
        # reimage
        im2 = rtlib.imgonefullxy(n.outer(u, d['freq']/d['freq_orig'][0]), n.outer(v, d['freq']/d['freq_orig'][0]), data_resamp[i0+candints[i]], d['npixx_full'], d['npixy_full'], d['uvres'], verbose=0)

        # find most extreme pixel
        snrmax = im2.max()/im2.std()
        snrmin = im2.min()/im2.std()
        if snrmax >= abs(snrmin):
            snr2 = snrmax
        else:
            snr2 = snrmin
        # threshold
        if abs(snr2) > d['sigma_image2']:
            # calc loc in first image
            if snr[i] > 0:
                peakl, peakm = n.where(ims[i] == ims[i].max())
            else:
                peakl, peakm = n.where(ims[i] == ims[i].min())
            l1 = (d['npixx']/2. - peakl[0])/(d['npixx']*d['uvres'])
            m1 = (d['npixy']/2. - peakm[0])/(d['npixy']*d['uvres'])

            # calc src loc in second image
            if snr2 > 0:
                peakl, peakm = n.where(im2 == im2.max())
            else:
                peakl, peakm = n.where(im2 == im2.min())
            l2 = (d['npixx_full']/2. - peakl[0])/(d['npixx_full']*d['uvres'])
            m2 = (d['npixy_full']/2. - peakm[0])/(d['npixy_full']*d['uvres'])
            logger.info('Got one!  Int=%d, DM=%d, dt=%d: SNR_im1=%.1f, SNR_im2=%.1f @ (%.2e,%.2e).' % ((i0+candints[i])*d['dtarr'][dtind], d['dmarr'][dmind], d['dtarr'][dtind], snr[i], snr2, l2, m2))
            candid =  (d['segment'], (i0+candints[i])*d['dtarr'][dtind], dmind, dtind, beamnum)

            # assemble feature in requested order
            ff = []
            for feature in d['features']:
                if feature == 'snr1':
                    ff.append(snr[i])
                elif feature == 'immax1':
                    if snr[i] > 0:
                        ff.append(ims[i].max())
                    else:
                        ff.append(ims[i].min())
                elif feature == 'l1':
                    ff.append(l1)
                elif feature == 'm1':
                    ff.append(m1)
                elif feature == 'snr2':
                    ff.append(snr2)
                elif feature == 'immax2':
                    if snr2 > 0:
                        ff.append(im2.max())
                    else:
                        ff.append(im2.min())
                elif feature == 'l2':
                    ff.append(l2)
                elif feature == 'm2':
                    ff.append(m2)

            feat[candid] = list(ff)
        else:
            logger.info('Almost...  Int=%d, DM=%d, dt=%d: SNR_im1=%.1f, SNR_im2=%.1f.' % ((i0+candints[i])*d['dtarr'][dtind], d['dmarr'][dmind], d['dtarr'][dtind], snr[i], snr2))

    return feat

def image2w(d, i0, i1, u, v, w, dmind, dtind, beamnum, bls, uvkers):
    """ Parallelizable function for imaging a chunk of data for a single dm.
    Assumes data is dedispersed and resampled, so this just images each integration.
    Two-stage imaging uses ideal uv coverage in second image.
    returns dictionary with keys of cand location and values as tuple of features
    """

    readints = len(data_resamp_mem)/(d['nbl']*d['nchan']*d['npol']*2)
    data_resamp = numpyview(data_resamp_mem, 'complex64', (readints, d['nbl'], d['nchan'], d['npol']))

    ims,snr,candints = rtlib.imgallfullfilterxy(n.outer(u, d['freq']/d['freq_orig'][0]), n.outer(v, d['freq']/d['freq_orig'][0]), data_resamp[i0:i1], d['npixx'], d['npixy'], d['uvres'], d['sigma_image1'])

    feat = {}
    for i in xrange(len(candints)):
        # reimage
        npix = max(d['npixx_full'], d['npixy_full'])
        im2 = rtlib.imgonefullw(n.outer(u, d['freq']/d['freq_orig'][0]), n.outer(v, d['freq']/d['freq_orig'][0]), data_resamp[i0+candints[i]], npix, d['uvres'], bls, uvkers, verbose=1)
        logger.debug(im2.shape)

        # find most extreme pixel
        snrmax = im2.max()/im2.std()
        snrmin = im2.min()/im2.std()
        if snrmax >= abs(snrmin):
            snr2 = snrmax
        else:
            snr2 = snrmin
        # threshold
        if abs(snr2) > d['sigma_image2']:
            # calc loc in first image
            if snr[i] > 0:
                peakl, peakm = n.where(ims[i] == ims[i].max())
            else:
                peakl, peakm = n.where(ims[i] == ims[i].min())
            l1 = (d['npixx']/2. - peakl[0])/(d['npixx']*d['uvres'])
            m1 = (d['npixy']/2. - peakm[0])/(d['npixy']*d['uvres'])

            # calc src loc in second image
            if snr2 > 0:
                peakl, peakm = n.where(im2 == im2.max())
            else:
                peakl, peakm = n.where(im2 == im2.min())
            l2 = (npix/2. - peakl[0])/(npix*d['uvres'])
            m2 = (npix/2. - peakm[0])/(npix*d['uvres'])
            logger.info('Got one!  Int=%d, DM=%d, dt=%d: SNR_im1=%.1f, SNR_im2=%.1f @ (%.2e,%.2e).' % ((i0+candints[i])*d['dtarr'][dtind], d['dmarr'][dmind], d['dtarr'][dtind], snr[i], snr2, l2, m2))
            candid =  (d['segment'], (i0+candints[i])*d['dtarr'][dtind], dmind, dtind, beamnum)

            # assemble feature in requested order
            ff = []
            for feature in d['features']:
                if feature == 'snr1':
                    ff.append(snr[i])
                elif feature == 'immax1':
                    if snr[i] > 0:
                        ff.append(ims[i].max())
                    else:
                        ff.append(ims[i].min())
                elif feature == 'l1':
                    ff.append(l1)
                elif feature == 'm1':
                    ff.append(m1)
                elif feature == 'snr2':
                    ff.append(snr2)
                elif feature == 'immax2':
                    if snr2 > 0:
                        ff.append(im2.max())
                    else:
                        ff.append(im2.min())
                elif feature == 'l2':
                    ff.append(l2)
                elif feature == 'm2':
                    ff.append(m2)

            feat[candid] = list(ff)
        else:
            logger.info('Almost...  Int=%d, DM=%d, dt=%d: SNR_im1=%.1f, SNR_im2=%.1f.' % ((i0+candints[i])*d['dtarr'][dtind], d['dmarr'][dmind], d['dtarr'][dtind], snr[i], snr2))

    return feat

def image1wrap(d, u, v, w, npixx, npixy, candint):
    """ Parallelizable function for imaging a chunk of data for a single dm.
    Assumes data is dedispersed and resampled, so this just images each integration.
    Simple one-stage imaging that returns dict of params.
    returns dictionary with keys of cand location and values as tuple of features
    """

    readints = len(data_resamp_mem)/(d['nbl']*d['nchan']*d['npol']*2)
    data_resamp = numpyview(data_resamp_mem, 'complex64', (readints, d['nbl'], d['nchan'], d['npol']))

    image = rtlib.imgonefullxy(n.outer(u, d['freq']/d['freq_orig'][0]), n.outer(v, d['freq']/d['freq_orig'][0]), data_resamp[candint], npixx, npixy, d['uvres'], verbose=1)
    return image

def sample_image(d, data, u, v, w, i=-1, verbose=1, imager='xy', wres=100):
    """ Samples one integration and returns image
    i is integration to image. Default is mid int.
    """

    if i == -1:
        i = len(data)/2

    if imager == 'xy':
        image = rtlib.imgonefullxy(n.outer(u, d['freq']/d['freq_orig'][0]), n.outer(v, d['freq']/d['freq_orig'][0]), data[i], d['npixx'], d['npixy'], d['uvres'], verbose=verbose)
    elif imager == 'w':
        npix = max(d['npixx'], d['npixy'])
        bls, uvkers = rtlib.genuvkernels(w, wres, npix, d['uvres'], ksize=21, oversample=1)
        image = rtlib.imgonefullw(n.outer(u, d['freq']/d['freq_orig'][0]), n.outer(v, d['freq']/d['freq_orig'][0]), data[i], npix, d['uvres'], bls, uvkers, verbose=verbose)

#        bls, lmkers = rtlib.genlmkernels(w, wres, npix, d['uvres'])
#        image = rtlib.imgonefullw(n.outer(u, d['freq']/d['freq_orig'][0]), n.outer(v, d['freq']/d['freq_orig'][0]), data[i], npix, d['uvres'], [bls[0]], [lmkers[0]], verbose=verbose)

    return image

def estimate_noiseperbl(data):
    """ Takes large data array and sigma clips it to find noise per bl for input to detect_bispectra.
    Takes mean across pols and channels for now, as in detect_bispectra.
    """
    
    # define noise per baseline for data seen by detect_bispectra or image
    datamean = data.mean(axis=2).imag                      # use imaginary part to estimate noise without calibrated, on-axis signal
    (datameanmin, datameanmax) = rtlib.sigma_clip(datamean.flatten())
    good = n.where( (datamean>datameanmin) & (datamean<datameanmax) )
    noiseperbl = datamean[good].std()   # measure single noise for input to detect_bispectra
    logger.info('Clipped to %d%% of data (%.3f to %.3f). Noise = %.3f.' % (100.*len(good[0])/len(datamean.flatten()), datameanmin, datameanmax, noiseperbl))
    return noiseperbl

def noisepickle(d, data, u, v, w, chunk=200):
    """ Calculates noise properties and saves values to pickle.
    chunk defines window for measurement. at least one measurement always made.
    """

    noisefile = 'noise_' + d['fileroot'] + '_sc' + str(d['scan']) + 'seg' + str(d['segment']) + '.pkl'

    if d['savenoise']:
        if os.path.exists(noisefile):
            logger.warn('noisefile %s already exists' % noisefile)
            return

    nints = len(data)
    chunk = min(chunk, nints-1)  # ensure at least one measurement
    results = []
    pkl = open(noisefile, 'a')
    for (i0, i1) in zip(range(0, nints-chunk, chunk), range(chunk, nints, chunk)):
        imid = (i0+i1)/2
        noiseperbl = estimate_noiseperbl(data[i0:i1])
        imstd = sample_image(d, data, u, v, w, imid, verbose=0).std()
        zerofrac = float(len(n.where(data[i0:i1] == 0j)[0]))/data[i0:i1].size
        results.append( (d['segment'], noiseperbl, zerofrac, imstd) )
    pickle.dump(results, pkl)
    pkl.close()

def savecands(d, cands):
    """ Save all candidates in pkl file for later aggregation and filtering.
    """

    candsfile = 'cands_' + d['fileroot'] + '_sc' + str(d['scan']) + 'seg' + str(d['segment']) + '.pkl'

    pkl = open(candsfile, 'w')
    pickle.dump(d, pkl)
    pickle.dump(cands, pkl)
    pkl.close()

def numpyview(arr, datatype, shape):
    """ Takes mp shared array and returns numpy array with given shape.
    """

#    return n.frombuffer(arr.get_obj(), dtype=n.dtype(datatype)).view(n.dtype(datatype)).reshape(shape)  # for shared mp.Array
    return n.frombuffer(arr, dtype=n.dtype(datatype)).view(n.dtype(datatype)).reshape(shape)   # for shared mps.RawArray

def initpool(shared_arr_):
    global data_resamp_mem
    data_resamp_mem = shared_arr_ # must be inhereted, not passed as an argument

def initpool2(shared_arr_):
    global data_mem
    data_mem = shared_arr_ # must be inhereted, not passed as an argument

