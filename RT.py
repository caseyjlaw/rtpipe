import rtpipe.parsems as pm
import rtpipe.parsecal as pc
import rtpipe.parsesdm as ps
import rtlib_cython as rtlib
import multiprocessing as mp
import multiprocessing.sharedctypes as mps
from contextlib import closing
import numpy as n
from scipy.special import erf
import casautil, os, pickle, glob

qa = casautil.tools.quanta()

def pipeline(d, segment, reproducecand=()):
    """ Transient search pipeline running on single node.
    Processes a single segment of data (where a single bgsub, (u,v,w), etc. can be used).
    Searches completely, independently, and saves candidates.
    reproducecand is tuple of (candint, dmind, dtind) that params to reproduce/visualize candidate

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

    if d['dataformat'] == 'ms':   # CASA-based read
        segread = pm.readsegment(d, segment)
        nints = len(segread[0])
        data = n.empty( (nints, d['nbl'], d['nchan'], d['npol']), dtype='complex64', order='C')
        data[:] = segread[0]
        (u, v, w) = (segread[1][nints/2], segread[2][nints/2], segread[3][nints/2])  # mid int good enough for segment. could extend this to save per chunk
        del segread
    elif d['dataformat'] == 'sdm':
        t0 = d['segmenttimes'][segment][0]
        t1 = d['segmenttimes'][segment][1]
        readints = n.round(24*3600*(t1 - t0)/d['inttime'], 0).astype(int)/d['read_downsample']

        # for shared mem
        data_mem = mps.RawArray(mps.ctypes.c_float, long(readints*d['nbl']*d['nchan']*d['npol'])*2)  # 'long' type needed to hold whole (2 min, 5 ms) scans
        data = numpyview(data_mem, 'complex64', (readints, d['nbl'], d['nchan'], d['npol']))
        # for non shared mem
#        data = n.empty( (readints, d['nbl'], d['nchan'], d['npol']), dtype='complex64', order='C')

        data[:] = ps.read_bdf_segment(d, segment)
        (u,v,w) = ps.get_uvw_segment(d, segment)
    else:
        print 'Data format %s not supported.' % d['dataformat']

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
        print 'Calibration file not found. Proceeding with no calibration applied.'

    # flag data
    dataflagpool(data_mem, d)

    # mean t vis subtration
    meantsubpool(data_mem, d)

    if d['savenoise']:
        noisepickle(d, data, u, v, w)      # save noise pickle

    ####    ####    ####    ####
    # 3) Search using all threads
    ####    ####    ####    ####
    print 'Starting search...'
    if len(reproducecand) == 0:
        cands = search(d, data, u, v, w)

        ####    ####    ####    ####
        # 4) Save candidate info
        ####    ####    ####    ####
        if d['savecands']:
            print 'Saving %d candidates...' % (len(cands))
            savecands(d, cands)

        return len(cands)

    elif len(reproducecand) == 3:  # reproduce and visualize candidates
        reproduceint, dmind, dtind = reproducecand
        d['dmarr'] = [d['dmarr'][dmind]]
        d['dtarr'] = [d['dtarr'][dtind]]
        im, data = reproduce(d, data, u, v, w, reproduceint)
        return im, data
    else:
        print 'reproducecand should be empty of length 3: %s' % reproducecand

def meantsubpool(data_mem, d):
    """ Parallelized mean t visibility subtraction.
    """

    if d['timesub'] == 'mean':
        print 'Subtracting mean visibility in time...'
        blranges = [(d['nbl'] * t/d['nthread'], d['nbl']*(t+1)/d['nthread']) for t in range(d['nthread'])]
        with closing(mp.Pool(d['nthread'], initializer=initpool2, initargs=(data_mem,))) as pool:
            for blr in blranges:
                pool.apply_async(meantsub, [blr, d])
    else:
        print 'No mean time subtraction.'

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


    if d['flagmode'] == 'standard':
        print 'Flagging data...'
        chperspw = len(d['freq_orig'])/len(d['spw'])
        with closing(mp.Pool(d['nthread'], initializer=initpool2, initargs=(data_mem,))) as pool:
            resultd = {}
            for pol in range(d['npol']):
                for spw in range(d['nspw']):
                    freqs = d['freq_orig'][spw*chperspw:(spw+1)*chperspw]  # find chans for spw. only works for 2 or more sb
                    chans = n.array([i for i in xrange(len(d['freq'])) if d['freq'][i] in freqs])
                    resultd[(spw,pol)] = pool.apply_async(dataflag, [chans, pol, d, 2.5, 'badch', 0.05])
            for kk in resultd.keys():
                result = resultd[kk].get()
#                print kk, result, result/(0.5*data.size)

            resultd = {}
            for spw in range(d['nspw']):
                freqs = d['freq_orig'][spw*chperspw:(spw+1)*chperspw]  # find chans for spw. only works for 2 or more sb
                chans = n.array([i for i in xrange(len(d['freq'])) if d['freq'][i] in freqs])
                resultd[spw] = pool.apply_async(dataflag, [chans, 0, d, 3., 'badap', 0.2]) # pol not used here
            for kk in resultd.keys():
                result = resultd[kk].get()
#                print kk, result, result/(0.5*data.size)

            resultd = {}
            for pol in range(d['npol']):
                for spw in range(d['nspw']):
                    freqs = d['freq_orig'][spw*chperspw:(spw+1)*chperspw]  # find chans for spw. only works for 2 or more sb
                    chans = n.array([i for i in xrange(len(d['freq'])) if d['freq'][i] in freqs])
                    resultd[(spw,pol)] = pool.apply_async(dataflag, [chans, pol, d, 4., 'blstd', 0.1])
            for kk in resultd.keys():
                result = resultd[kk].get()
#                print kk, result, result/(0.5*data.size)

            resultd = {}
            for pol in range(d['npol']):
                for spw in range(d['nspw']):
                    freqs = d['freq_orig'][spw*chperspw:(spw+1)*chperspw]  # find chans for spw. only works for 2 or more sb
                    chans = n.array([i for i in xrange(len(d['freq'])) if d['freq'][i] in freqs])
                    resultd[(spw,pol)] = pool.apply_async(dataflag, [chans, pol, d, 4., 'ring', 0.2])
            for kk in resultd.keys():
                result = resultd[kk].get()
#                print kk, result, result/(0.5*data.size)

    else:
        print 'No real-time flagging.'

def dataflag(chans, pol, d, sig, mode, conv):
    """ Wrapper function to get shared memory as numpy array into pool
    Assumes data_mem is global mps.RawArray
    """
    readints = len(data_mem)/(d['nbl']*d['nchan']*d['npol']*2)
    data = numpyview(data_mem, 'complex64', (readints, d['nbl'], d['nchan'], d['npol']))

    return rtlib.dataflag(data, chans, pol, d, sig, mode, conv)

def search(d, data, u, v, w):
    """ Search function.
    Queues all trials with multiprocessing.
    Assumes shared memory system with single uvw grid for all images.
    """

    beamnum = 0
    cands = {}
    resultlist = []

    if d['savecands']:
        candsfile = 'cands_' + d['fileroot'] + '_sc' + str(d['scan']) + 'seg' + str(d['segment']) + '.pkl'
        if os.path.exists(candsfile):
            print 'candsfile %s already exists' % candsfile
            return cands

    nints = len(data)
#    data_resamp_mem = [mps.RawArray(mps.ctypes.c_float, (nints*d['nbl']*d['nchan']*d['npol'])*2) for resamp in range(len(d['dtarr']))]    # simple shared array
#    data_resamp = [numpyview(data_resamp_mem[resamp], 'complex64', (nints, d['nbl'], d['nchan'], d['npol'])) for resamp in range(len(d['dtarr']))]
#    data_resamp[0][:] = data[:]
#    del data
    data_resamp_mem = mps.RawArray(mps.ctypes.c_float, (nints*d['nbl']*d['nchan']*d['npol'])*2)
    data_resamp = numpyview(data_resamp_mem, 'complex64', (nints, d['nbl'], d['nchan'], d['npol']))

    # make wterm kernels
    if d['searchtype'] == 'image2w':
        wres = 100
        npix = max(d['npixx_full'], d['npixy_full'])
        bls, uvkers = rtlib.genuvkernels(w, wres, npix, d['uvres'], thresh=0.05)

    # SUBMITTING THE LOOPS
    if n.any(data):
        print 'Searching in %d chunks with %d threads' % (d['nchunk'], d['nthread'])

        # open pool to run jobs
        print 'Dedispering to max (DM, dt) of (%d, %d) ...' % (d['dmarr'][-1], d['dtarr'][-1]), 
        for dmind in xrange(len(d['dmarr'])):
            for dtind in xrange(len(d['dtarr'])):
                print '(%d,%d)' % (d['dmarr'][dmind], d['dtarr'][dtind]),
                data_resamp[:] = data[:]
                blranges = [(d['nbl'] * t/d['nthread'], d['nbl']*(t+1)/d['nthread']) for t in range(d['nthread'])]
                with closing(mp.Pool(d['nthread'], initializer=initpool, initargs=(data_resamp_mem,))) as pool:
                    for blr in blranges:
                        result = pool.apply_async(correct_dmdt, [d, dmind, dtind, blr])
                        resultlist.append(result)
                    for result in resultlist:
                        result.wait()
                    resultlist = []

                    print 'Imaging...',
                    for chunk in range(d['nchunk']):
                        i0 = (nints/d['dtarr'][dtind])*chunk/d['nchunk']
                        i1 = (nints/d['dtarr'][dtind])*(chunk+1)/d['nchunk']
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
        print 'Data for processing is zeros. Moving on...'

    print 'Found %d cands in scan %d segment %d of %s. ' % (len(cands), d['scan'], d['segment'], d['filename'])
    return cands

def reproduce(d, data, u, v, w, candint, twindow=30):
    """ Reproduce function, much like search.
    Instead of returning cand count
    Assumes shared memory system with single uvw grid for all images.
    """

    nints = len(data)
    data_resamp_mem = mps.RawArray(mps.ctypes.c_float, (nints*d['nbl']*d['nchan']*d['npol'])*2)
    data_resamp = [numpyview(data_resamp_mem, 'complex64', (nints, d['nbl'], d['nchan'], d['npol']))]
    data_resamp[:] = data[:]
    del data

    dmind = 0; dtind = 0
    with closing(mp.Pool(1, initializer=initpool, initargs=(data_resamp_mem,))) as pool:
        # dedisperse
        print 'Dedispersing with DM=%.1f, dt=%d...' % (d['dmarr'][dmind], d['dtarr'][dtind])
        pool.apply(correct_dmdt, [d, dmind, dtind])

        # set up image
        if d['searchtype'] == 'image1':
            npixx = d['npixx']
            npixy = d['npixy']
        elif d['searchtype'] == 'image2':
            npixx = d['npixx_full']
            npixy = d['npixy_full']

        # image
        print 'Imaging int %d with %d %d pixels...' % (candint, npixx, npixy)
#        print nints/d['dtarr'][dtind], candint/d['dtarr'][dtind]
#        ims,snrs,candints = pool.apply(rtlib.imgallfullfilterxyflux, [n.outer(u, d['freq']/d['freq_orig'][0]), n.outer(v, d['freq']/d['freq_orig'][0]), data_resamp[0][0:nints/d['dtarr'][dtind]], npixx, npixy, d['uvres'], d['sigma_image1']])
#        print snrs, candints
#        im = ims[candints.index(candint)]
        im = pool.apply(rtlib.imgonefullxy, [n.outer(u, d['freq']/d['freq_orig'][0]), n.outer(v, d['freq']/d['freq_orig'][0]), data_resamp[candint/d['dtarr'][dtind]], npixx, npixy, d['uvres'], 0])

        print 'Made image with SNR min, max: %.1f, %.1f' % (im.min()/im.std(), im.max()/im.std())
        peakl, peakm = n.where(im == im.max())
        l1 = (float((npixx)/d['uvres'])/2. - peakl[0])/npixx
        m1 = (float((npixy)/d['uvres'])/2. - peakm[0])/npixy

        # rephase and trim interesting ints out
        print 'Rephasing to peak...'
        pool.apply(move_phasecenter, [d, l1, m1, u, v])
        minint = max(candint-twindow/2, 0)
        maxint = min(candint+twindow/2, len(data_resamp))
        data = data_resamp[minint:maxint].copy()

    return(im, data.mean(axis=1))

def set_pipeline(filename, scan, fileroot='', paramfile='', **kwargs):
    """ Function defines pipeline state for search. Takes data/scan as input.
    fileroot is base name for associated products (cal files, noise, cands). if blank, it is set to filename.
    paramfile is name of file that defines all pipeline parameters (python-like syntax).
    kwargs used to overload paramfile definitions.
    Many parameters take 0 as default, which auto-defines ideal parameters. 
    This definition does not yet consider memory/cpu/time limitations.
    nsegments defines how to break jobs in time. nchunk defines how each segment is split in time for imaging.
    """

    # define metadata (state) dict. chans/spw is special because it goes in to get_metadata call
    if 'chans' in kwargs.keys(): 
        chans=kwargs['chans']
    else:
        chans = []
    if 'spw' in kwargs.keys(): 
        spw=kwargs['spw']
    else:
        spw = []

    # then get all metadata
    if os.path.exists(os.path.join(filename, 'Antenna.xml')):
        d = ps.get_metadata(filename, scan, chans=chans, spw=spw, params=paramfile)   # can take file name or Params instance
        d['dataformat'] = 'sdm'
    else:
        d = pm.get_metadata(filename, scan, chans=chans, spw=spw, params=paramfile)
        d['dataformat'] = 'ms'

    # overload with provided kwargs
    for key in kwargs.keys():
        print 'Setting %s to %s' % (key, kwargs[key])
        d[key] = kwargs[key]

    # define rootname for in/out cal/products
    if fileroot:
        d['fileroot'] = fileroot
    else:
        d['fileroot'] = os.path.split(filename)[-1]

    # autodetect calibration products
    if not d['gainfile']:
        filelist = glob.glob(os.path.join(d['workdir'], d['fileroot'] + '.g?'))
        if len(filelist):
            filelist.sort()
            d['gainfile'] = filelist[-1]
            print 'Autodetecting cal files... gainfile set to %s.' % d['gainfile']
    if not d['bpfile']:
        filelist = glob.glob(os.path.join(d['workdir'], d['fileroot'] + '.b?'))
        if len(filelist):
            filelist.sort()
            d['bpfile'] = filelist[-1]
            print 'Autodetecting cal files... bpfile set to %s.' % d['bpfile']

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
    if d['npix'] == 0:
        d['npixx'] = d['npixx_full']
        d['npixy'] = d['npixy_full']
        d['npix'] = max(d['npixx'], d['npixy'])
    else:
        d['npixx'] = d['npix']
        d['npixy'] = d['npix']

    # define times for data to read
    d['t_overlap'] = rtlib.calc_delay(d['freq'], d['inttime'], max(d['dmarr'])).max()*d['inttime']   # time of overlap for total dm coverage at segment boundaries
    d['datadelay'] = n.max([rtlib.calc_delay(d['freq'], d['inttime'],dm).max() for dm in d['dmarr']])
    d['nints'] = d['nints'] - d['nskip']

    if d['nsegments'] == 0:
        fringetime = calc_fringetime(d)
        stopdts = n.arange(d['nskip']*d['inttime']+d['t_overlap'], d['nints']*d['inttime'], fringetime-d['t_overlap'])[1:]
        startdts = n.concatenate( ([d['nskip']*d['inttime']], stopdts[:-1]-d['t_overlap']) )
        d['nsegments'] = len(startdts)
    else:
        stopdts = n.linspace(d['nskip']*d['inttime']+d['t_overlap'], d['nints']*d['inttime'], d['nsegments']+1)[1:]
        startdts = n.concatenate( ([d['nskip']*d['inttime']], stopdts[:-1]-d['t_overlap']) )

    segmenttimes = []
    for (startdt, stopdt) in zip(startdts, stopdts):
        starttime = qa.getvalue(qa.convert(qa.time(qa.quantity(d['starttime_mjd']+startdt/(24.*60*60),'d'),form=['ymd'], prec=9)[0], 's'))[0]/(24*3600)
        stoptime = qa.getvalue(qa.convert(qa.time(qa.quantity(d['starttime_mjd']+stopdt/(24.*60*60), 'd'), form=['ymd'], prec=9)[0], 's'))[0]/(24*3600)
        segmenttimes.append((starttime, stoptime))
    d['segmenttimes'] = n.array(segmenttimes)
    d['t_segment'] = 24*3600*(d['segmenttimes'][0,1]-d['segmenttimes'][0,0])

    # farm out work in pieces to use all threads well
    if d['nchunk'] == 0:
        d['nchunk'] = d['nthread']

    # scaling of number of integrations beyond dt=1
    dtfactor = n.sum([1./i for i in d['dtarr']])    # assumes dedisperse-all algorithm

    # calculate number of thermal noise candidates per segment
    ntrials = d['t_segment']/(d['inttime']*d['read_downsample']) * dtfactor * len(d['dmarr']) * d['npixx'] * d['npixy']
    qfrac = 1 - (erf(d['sigma_image1']/n.sqrt(2)) + 1)/2.
    nfalse = int(qfrac*ntrials)

    print 'Pipeline summary:'
    print '\t Products saved with %s. Calibration files set to (%s, %s)' % (d['fileroot'], d['gainfile'], d['bpfile'])
    print '\t Using %d segment%s of %d ints (%.1f s) with overlap of %.1f s' % (d['nsegments'], "s"[not d['nsegments']-1:], d['t_segment']/(d['inttime']*d['read_downsample']), d['t_segment'], d['t_overlap'])
    if d['t_overlap'] > d['t_segment']/3.:
        print '\t\t **Inefficient search: Max DM sweep (%.1f s) close to segment size (%.1f s)**' % (d['t_overlap'], d['t_segment'])
    print '\t Downsampling by %d and skipping %d ints from start of scan.' % (d['read_downsample'], d['nskip'])
    print '\t Excluding ants %s' % (d['excludeants'])
    print

    print '\t Search with %s and threshold %.1f.' % (d['searchtype'], d['sigma_image1'])
    print '\t Using uvgrid npix=(%d,%d) and res=%d.' % (d['npixx'], d['npixy'], d['uvres'])
    print '\t Expect %d thermal false positives per segment.' % nfalse

    print
    vismem0 = (8*(d['t_segment']/(d['inttime']*d['read_downsample']) * d['nbl'] * d['nchan'] * d['npol'])/1024**3)
    print '\t Visibility memory usage is %d GB/segment' % vismem0 * dtfactor
    print '\t Imaging in %d chunk%s using max of %d GB/segment' % (d['nchunk'], "s"[not d['nsegments']-1:], 8*(d['t_segment']/(d['inttime']*d['read_downsample']) * d['npixx'] * d['npixy'])/1024**3)
    print '\t Grand total memory usage: %d GB/segment' % ( (vismem0 * dtfactor) + 8*(d['t_segment']/(d['inttime']*d['read_downsample']) * d['npixx'] * d['npixy'])/1024**3)

    return d

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

def move_phasecenter(d, l1, m1, u, v):
    """ Handler function for phaseshift_threaded
    """

    datadt = data_resamp[0]
    rtlib.phaseshift_threaded(datadt, d, l1, m1, u, v)

def calc_dmgrid(d, maxloss=0.05, dt=3000., mindm=0., maxdm=2000.):
    """ Function to calculate the DM values for a given maximum sensitivity loss.
    maxloss is sensitivity loss tolerated by dm bin width. dt is assumed pulse width.
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
        print 'Got one!  Int=%d, DM=%d, dt=%d: SNR_im=%.1f @ (%.2e,%.2e).' % (d['nskip']+(i0+candints[i])*d['dtarr'][dtind], d['dmarr'][dmind], d['dtarr'][dtind], snr[i], l1, m1)
        candid =  (d['segment'], d['nskip']+(i0+candints[i])*d['dtarr'][dtind], dmind, dtind, beamnum)

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
            print 'Got one!  Int=%d, DM=%d, dt=%d: SNR_im1=%.1f, SNR_im2=%.1f @ (%.2e,%.2e).' % (d['nskip']+(i0+candints[i])*d['dtarr'][dtind], d['dmarr'][dmind], d['dtarr'][dtind], snr[i], snr2, l2, m2)
            candid =  (d['segment'], d['nskip']+(i0+candints[i])*d['dtarr'][dtind], dmind, dtind, beamnum)

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
            print 'Almost...  Int=%d, DM=%d, dt=%d: SNR_im1=%.1f, SNR_im2=%.1f.' % (d['nskip']+(i0+candints[i])*d['dtarr'][dtind], d['dmarr'][dmind], d['dtarr'][dtind], snr[i], snr2)

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
        print im2.shape

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
            print 'Got one!  Int=%d, DM=%d, dt=%d: SNR_im1=%.1f, SNR_im2=%.1f @ (%.2e,%.2e).' % (d['nskip']+(i0+candints[i])*d['dtarr'][dtind], d['dmarr'][dmind], d['dtarr'][dtind], snr[i], snr2, l2, m2)
            candid =  (d['segment'], d['nskip']+(i0+candints[i])*d['dtarr'][dtind], dmind, dtind, beamnum)

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
            print 'Almost...  Int=%d, DM=%d, dt=%d: SNR_im1=%.1f, SNR_im2=%.1f.' % (d['nskip']+(i0+candints[i])*d['dtarr'][dtind], d['dmarr'][dmind], d['dtarr'][dtind], snr[i], snr2)

    return feat

def sample_image(d, data, u, v, w, i=-1, verbose=1, imager='xy', wres=100):
    """ Samples one integration and returns image
    i is integration to image. Default is mid int.
    """

    if i == -1:
        i = len(data)/2

    if imager == 'xy':
        image = rtlib.imgonefullxy(n.outer(u, d['freq']/d['freq_orig'][0]), n.outer(v, d['freq']/d['freq_orig'][0]), data[i], d['npixx'], d['npixy'], d['uvres'], verbose=verbose)
    elif imager == 'w':
        bls, uvkers = rtlib.genuvkernels(w, wres, d['npix'], d['uvres'], thresh=0.05)
        image = rtlib.imgonefullw(n.outer(u, d['freq']/d['freq_orig'][0]), n.outer(v, d['freq']/d['freq_orig'][0]), data[i], d['npix'], d['uvres'], bls, uvkers, verbose=verbose)

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
    print 'Clipped to %d%% of data (%.3f to %.3f). Noise = %.3f.' % (100.*len(good[0])/len(datamean.flatten()), datameanmin, datameanmax, noiseperbl)
    return noiseperbl

def noisepickle(d, data, u, v, w, chunk=200):
    """ Calculates noise properties and saves values to pickle.
    chunk defines window for measurement. at least one measurement always made.
    """

    noisefile = 'noise_' + d['fileroot'] + '_sc' + str(d['scan']) + 'seg' + str(d['segment']) + '.pkl'

    if d['savenoise']:
        if os.path.exists(noisefile):
            print 'noisefile %s already exists' % noisefile
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
