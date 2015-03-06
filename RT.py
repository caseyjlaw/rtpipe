import realtime.parsems as pm
import realtime.parsecal as pc
import realtime.parsesdm as ps
import rtlib_cython as rtlib
import multiprocessing as mp
import multiprocessing.sharedctypes as mps
from contextlib import closing
import numpy as n
from scipy.special import erf
import casautil, os, pickle

qa = casautil.tools.quanta()

global data
global data_resamp

def pipeline(d, segment, msdata=False):
    """ Transient search pipeline running on single node.
    Processes a single segment of data (where a single bgsub, (u,v,w), etc. can be used).
    Searches completely, independently, and saves candidates.

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
    
    if msdata:   # CASA-based read
        segread = pm.readsegment(d, segment)
        nints = len(segread[0])
        data = n.empty( (nints, d['nbl'], d['nchan'], d['npol']), dtype='complex64', order='C')
        data[:] = segread[0]
        (u, v, w) = (segread[1][nints/2], segread[2][nints/2], segread[3][nints/2])  # mid int good enough for segment. could extend this to save per chunk
        del segread
    else:
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

    ####    ####    ####    ####
    # 2) Prepare data
    ####    ####    ####    ####

    # calibrate data
    if d['gainfile']:
        print 'Applying CASA calibration...'
        sols = pc.casa_sol(d['gainfile'], flagants=d['flagantsol'])
        sols.parsebp(d['bpfile'])
        sols.setselection(d['segmenttimes'][segment].mean(), d['freq']*1e9)
        sols.apply(data, d['blarr'])

    # flag data
    dataflagall(data, d)

    if d['timesub'] == 'mean':
        print 'Subtracting mean visibility in time...'
        rtlib.meantsub(data)
    else:
        print 'No mean time subtraction.'
#        data = data.mean(axis=0)[None,:,:,:]  # test

    if d['noisefile']:
        noisepickle(d, data, u, v, w)      # save noise pickle

    ####    ####    ####    ####
    # 3) Search using all threads
    ####    ####    ####    ####
    print 'Starting search...'
    cands = search(d, data, u, v, w)

    ####    ####    ####    ####
    # 4) Save candidate info
    ####    ####    ####    ####
    if d['candsfile']:
        print 'Saving %d candidates...' % (len(cands))
        savecands(d, cands)

    return len(cands)

def dataflag(chans, pol, d, sig, mode, conv):
    return rtlib.dataflag(data, chans, pol, d, sig, mode, conv)

def dataflagall(data, d):
    """ Parallelized flagging
    """

    if d['flagmode'] == 'standard':
        print 'Flagging data...'
        chperspw = len(d['freq_orig'])/len(d['spw'])
        with closing(mp.Pool(d['nthread'], initializer=initpool2, initargs=(data,))) as pool:
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

def search(d, data, u, v, w):
    """ Search function.
    Queues all trials with multiprocessing.
    Assumes shared memory system with single uvw grid for all images.
    """

    beamnum = 0
    cands = {}
    resultlist = []

    nints = len(data)
    data_resamp_mem = [mps.RawArray(mps.ctypes.c_float, (nints*d['nbl']*d['nchan']*d['npol'])*2) for resamp in range(len(d['dtarr']))]    # simple shared array
    data_resamp = [numpyview(data_resamp_mem[resamp], 'complex64', (nints, d['nbl'], d['nchan'], d['npol'])) for resamp in range(len(d['dtarr']))]
    data_resamp[0][:] = data[:]
    del data

    # make wterm kernels
    if d['searchtype'] == 'image2w':
        wres = 100
        npix = max(d['npixx_full'], d['npixy_full'])
        bls, uvkers = rtlib.genuvkernels(w, wres, npix, d['uvres'], thresh=0.05)

    # SUBMITTING THE LOOPS
    if n.any(data_resamp[0]):
        print 'Searching in %d chunks with %d threads' % (d['nchunk'], d['nthread'])

        # open pool to run jobs
        with closing(mp.Pool(d['nthread'], initializer=initpool, initargs=(data_resamp,))) as pool:
            for dmind in xrange(len(d['dmarr'])):
                print 'Dedispersing DM = %d (max %d)' % (d['dmarr'][dmind], d['dmarr'][-1])
                for dtind in xrange(len(d['dtarr'])):
                    data_resamp[dtind][:] = data_resamp[0][:]
# failed attempt to parallelize dedispersion by channel
#                    for chunk in range(d['nchunk']):
#                        ch0 = d['nchan']*chunk/d['nchunk']
#                        ch1 = d['nchan']*(chunk+1)/d['nchunk']
#                        result = pool.apply_async(correct_dmdt, [d, ch0, ch1, dmind, dtind])
                    result = pool.apply_async(correct_dmdt, [d, dmind, dtind])
                    resultlist.append(result)
                for result in resultlist:
                    result.wait()
                resultlist = []

                print 'Imaging DM = %d (max %d)' % (d['dmarr'][dmind], d['dmarr'][-1])
                for dtind in xrange(len(d['dtarr'])):
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
        pool.join()
    else:
        print 'Data for processing is zeros. Moving on...'

    print 'Found %d cands in segment %d of %s. ' % (len(cands), d['segment'], d['filename'])
    return cands

def set_pipeline(d, nskip=0, excludeants=[], dmarr=[0], dtarr=[1], uvres=0, npix=0, nsegments=0, nthread=16, nchunk=0, iterating=False, sigma_image1=6, timesub='', read_downsample=1, **kwargs):
    """ Function takes metadata (from parsems or parseasdm) and search parameters to define pipeline.
    iterating mode not yet supported.
    If nsegments=0, then the optimal size will be calculated based on uv extent.
    nchunk defines how segment is split in time for imaging. Will be limited by available memory.
    """

    d['nskip'] = nskip
    d['excludeants'] = excludeants
    d['dmarr'] = dmarr
    d['dtarr'] = dtarr
    d['nthread'] = nthread
    d['l0'] = 0.; d['m0'] = 0.
    d['timesub'] = timesub
    assert read_downsample > 0
    d['read_downsample'] = read_downsample
    d['sigma_image1'] = sigma_image1
    d['sigma_image2'] = sigma_image1
    d['flagmode'] = ''
    d['flagantsol'] = True
    d['gainfile'] = ''
    d['bpfile'] = ''
    d['noisefile'] = ''
    d['candsfile'] = ''
    d['searchtype'] = 'image1'
    # add in provided kwargs
    for key in kwargs.keys():
        print 'Setting %s to %s' % (key, kwargs[key])
        d[key] = kwargs[key]
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
    if uvres == 0:
        d['uvres'] = d['uvres_full']
    else:
        d['uvres'] = uvres
    if npix == 0:
        d['npixx'] = d['npixx_full']
        d['npixy'] = d['npixy_full']
    else:
        d['npixx'] = npix
        d['npixy'] = npix
    d['npix'] = max(d['npixx'], d['npixy'])

    # define times for data to read
    d['t_overlap'] = rtlib.calc_delay(d['freq'], d['inttime'], max(d['dmarr'])).max()*d['inttime']   # time of overlap for total dm coverage at segment boundaries
    d['datadelay'] = n.max([rtlib.calc_delay(d['freq'], d['inttime'],dm).max() for dm in d['dmarr']])
    d['nints'] = d['nints'] - d['nskip']

    if iterating:  # not supported yet
        if nsegments > 0:
            d['iterint'] = d['nints']/nsegments   # will likely need some logic here governed by memory/cpu resources
    else:
        if nsegments == 0:
            nsegments = calc_nsegments(d)
        d['nsegments'] = nsegments

        stopdts = n.linspace(d['nskip']*d['inttime']+d['t_overlap'], (d['nints'])*d['inttime'], d['nsegments']+1)[1:]
        startdts = n.concatenate( ([d['nskip']*d['inttime']], stopdts[:-1]-d['t_overlap']) )
        segmenttimes = []
        for (startdt, stopdt) in zip(startdts, stopdts):
            starttime = qa.getvalue(qa.convert(qa.time(qa.quantity(d['starttime_mjd']+startdt/(24.*60*60),'d'),form=['ymd'], prec=9)[0], 's'))[0]/(24*3600)
            stoptime = qa.getvalue(qa.convert(qa.time(qa.quantity(d['starttime_mjd']+stopdt/(24.*60*60), 'd'), form=['ymd'], prec=9)[0], 's'))[0]/(24*3600)
            segmenttimes.append((starttime, stoptime))
        d['segmenttimes'] = n.array(segmenttimes)
        d['t_segment'] = 24*3600*(d['segmenttimes'][0,1]-d['segmenttimes'][0,0])

    # check on cal files, if defined
    if d['gainfile']:
        if not os.path.exists(d['gainfile']):
            print 'Can\'t find gainfile %s' % d['gainfile']
    if d['bpfile']:
        if not os.path.exists(d['bpfile']):
            print 'Can\'t find bpfile %s' % d['bpfile']

    # farm out work in pieces to use all threads well
    if nchunk == 0:
        d['nchunk'] = d['nthread']
    else:
        d['nchunk'] = nchunk

    # scaling of number of integrations beyond dt=1
    dtfactor = n.sum([1./i for i in d['dtarr']])    # assumes dedisperse-all algorithm

    # calculate number of thermal noise candidates per segment
    ntrials = d['t_segment']/(d['inttime']*d['read_downsample']) * dtfactor * len(d['dmarr']) * d['npixx'] * d['npixy']
    qfrac = 1 - (erf(d['sigma_image1']/n.sqrt(2)) + 1)/2.
    nfalse = int(qfrac*ntrials)

    print 'Pipeline summary:'
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

def calc_nsegments(d):
    """ Estimate largest time span of a "segment".
    A segment is the maximal time span that can be have a single bg fringe subtracted and uv grid definition.
    Max fringe window estimated for 5% amp loss at first null averaged over all baselines. Assumes dec=+90, which is conservative.
    Returns time in seconds that defines good window.
    """

    maxbl = d['uvres']*d['npix']/2    # fringe time for imaged data only
    fringe_time = 0.5*(24*3600)/(2*n.pi*maxbl/25.)   # max fringe window in seconds
    nseg = max(int(d['nints']*d['inttime']/fringe_time), 1)   # minimum of 1 segment
    return nseg

def correct_dmdt(d, dmind, dtind):
    """ Dedisperses and resamples data *in place*.
    Drops edges, since it assumes that data is read with overlapping chunks in time.
    """

    datadt = data_resamp[dtind]
    dt = d['dtarr'][dtind]
    rtlib.dedisperse_resample(datadt, d['freq'], d['inttime'], d['dmarr'][dmind], dt, verbose=0)        # dedisperses data.
#    rtlib.dedisperse_resample(data[:,:,i0:i1], d['freq'][i0:i1], d['inttime'], d['dmarr'][dmind], dt, verbose=0)        # seems to not dedisperse

def image1(d, i0, i1, u, v, w, dmind, dtind, beamnum):
    """ Parallelizable function for imaging a chunk of data for a single dm.
    Assumes data is dedispersed and resampled, so this just images each integration.
    Simple one-stage imaging that returns dict of params.
    returns dictionary with keys of cand location and values as tuple of features
    """

    ims,snr,candints = rtlib.imgallfullfilterxyflux(n.outer(u, d['freq']/d['freq_orig'][0]), n.outer(v, d['freq']/d['freq_orig'][0]), data_resamp[dtind][i0:i1], d['npixx'], d['npixy'], d['uvres'], d['sigma_image1'])

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

    ims,snr,candints = rtlib.imgallfullfilterxy(n.outer(u, d['freq']/d['freq_orig'][0]), n.outer(v, d['freq']/d['freq_orig'][0]), data_resamp[dtind][i0:i1], d['npixx'], d['npixy'], d['uvres'], d['sigma_image1'])

    feat = {}
    for i in xrange(len(candints)):
        # reimage
        im2 = rtlib.imgonefullxy(n.outer(u, d['freq']/d['freq_orig'][0]), n.outer(v, d['freq']/d['freq_orig'][0]), data_resamp[dtind][i0+candints[i]], d['npixx_full'], d['npixy_full'], d['uvres'], verbose=0)

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

    ims,snr,candints = rtlib.imgallfullfilterxy(n.outer(u, d['freq']/d['freq_orig'][0]), n.outer(v, d['freq']/d['freq_orig'][0]), data_resamp[dtind][i0:i1], d['npixx'], d['npixy'], d['uvres'], d['sigma_image1'])

    feat = {}
    for i in xrange(len(candints)):
        # reimage
        npix = max(d['npixx_full'], d['npixy_full'])
        im2 = rtlib.imgonefullw(n.outer(u, d['freq']/d['freq_orig'][0]), n.outer(v, d['freq']/d['freq_orig'][0]), data_resamp[dtind][i0+candints[i]], npix, d['uvres'], bls, uvkers, verbose=1)
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

def numpyview(arr, datatype, shape):
    """ Takes mp.Array and returns numpy array with given shape.
    """

#    return n.frombuffer(arr.get_obj(), dtype=n.dtype(datatype)).view(n.dtype(datatype)).reshape(shape)  # for shared mp.Array
    return n.frombuffer(arr, dtype=n.dtype(datatype)).view(n.dtype(datatype)).reshape(shape)   # for shared mps.RawArray

def initpool(shared_arr_):
    global data_resamp
    data_resamp = shared_arr_ # must be inhereted, not passed as an argument

def initpool2(shared_arr_):
    global data
    data = shared_arr_ # must be inhereted, not passed as an argument

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

    noisefile = d['noisefile'].rstrip('.pkl') + '_sc' + str(d['scan']) + 'seg' + str(d['segment']) + '.pkl'

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

def savecands(d, cands):
    """ Save all candidates in pkl file for later aggregation and filtering.
    """

    candsfile = d['candsfile'].rstrip('.pkl') + '_sc' + str(d['scan']) + 'seg' + str(d['segment']) + '.pkl'

    pkl = open(candsfile, 'w')
    pickle.dump(d, pkl)
    pickle.dump(cands, pkl)
    pkl.close()

