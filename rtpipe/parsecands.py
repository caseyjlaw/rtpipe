from scipy.special import erfinv
import glob, os, logging, sys
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cPickle as pickle
import json, requests


def read_candidates(candsfile, snrmin=0, snrmax=999, returnstate=False):
    """ Reads candidate file and returns data as python object.

    candsfile is pkl file (for now) with (1) state dict and (2) cands object.
    cands object can either be a dictionary or tuple of two numpy arrays.
    
    Return tuple of two numpy arrays (location, properties).
    returned values can be filtered by snrmin and snrmax (on absolute value).
    returnstate will instead return (loc, prop, state).
    """

    # read in pickle file of candidates
    try:
        with open(candsfile, 'rb') as pkl:
            d = pickle.load(pkl)
            cands = pickle.load(pkl)
    except IOError:
        logger.error('Trouble parsing candsfile')
        loc = np.array([])
        prop = np.array([])
        if returnstate:
            return (loc, prop, d)
        else:
            return (loc, prop)

    if 'snr2' in d['features']:
        snrcol = d['features'].index('snr2')
    elif 'snr1' in d['features']:
        snrcol = d['features'].index('snr1')

    if isinstance(cands, dict):
        loc = []; prop = []
        for kk in sorted(cands.keys()):
            if ((np.abs(cands[kk][snrcol]) > snrmin) and (np.abs(cands[kk][snrcol]) < snrmax)):
                loc.append( list(kk) )
                prop.append( list(cands[kk]) )
        loc = np.array(loc)
        prop = np.array(prop)
    elif isinstance(cands, tuple):
        loc, prop = cands
        assert isinstance(loc, np.ndarray) and isinstance(prop, np.ndarray), 'if cands object is tuple, contents must be two ndarrays'
        snrsel = np.where( (np.abs(prop[:, snrcol]) > snrmin) & (np.abs(prop[:, snrcol]) < snrmax) )
        loc = loc[snrsel]
        prop = prop[snrsel]
    else:
        logger.error('Cands object (in cands file) must be dict or tuple(np.array, np.array).')

    logger.info('Read %d candidates from %s.' % (len(loc), candsfile))

    if returnstate:
        return loc, prop, d
    else:
        return loc, prop


def read_noise(noisefile):
    """ Function to read a noise file and parse columns.
    Works with both per-scan and merged noise files.
    """

    noises = pickle.load(open(noisefile, 'r'))

    scan = []; seg = []; noiseperbl = []; flagfrac = []; imnoise = []
    if len(noises[0]) == 4:
        for noise in noises:
            seg.append(noise[0]); noiseperbl.append(noise[1])
            flagfrac.append(noise[2]); imnoise.append(noise[3])
        return (np.array(seg), np.array(noiseperbl), np.array(flagfrac), np.array(imnoise))
    elif len(noises[0]) == 5:
        for noise in noises:
            scan.append(noise[0])
            seg.append(noise[1]); noiseperbl.append(noise[2])
            flagfrac.append(noise[3]); imnoise.append(noise[4])
        return (np.array(scan), np.array(seg), np.array(noiseperbl), np.array(flagfrac), np.array(imnoise))
    else:
        logger.warn('structure of noise file not understood. first entry should be length 4 of 5.')


def merge_segments(filename, scan, cleanup=True, sizelimit=0):
    """ Merges cands/noise pkl files from multiple segments to single cands/noise file.

    Expects segment cands pkls with have (1) state dict and (2) cands dict.
    Writes tuple state dict and duple of numpy arrays
    A single pkl written per scan using root name fileroot.
    if cleanup, it will remove segments after merging.
    if sizelimit, it will reduce the output file to be less than this many MB.
    """

    workdir = os.path.dirname(filename)
    fileroot = os.path.basename(filename)
    candslist = glob.glob(os.path.join(workdir, 'cands_' + fileroot + '_sc' + str(scan) + 'seg*.pkl'))
    noiselist = glob.glob(os.path.join(workdir, 'noise_' + fileroot + '_sc' + str(scan) + 'seg*.pkl'))
    candssegs = sorted([candsfile.rstrip('.pkl').split('seg')[1] for candsfile in candslist])
    noisesegs = sorted([noisefile.rstrip('.pkl').split('seg')[1] for noisefile in noiselist])

    # test for good list with segments
    if not candslist and not noiselist:
        logger.warn('candslist and noiselist are empty.')
        return

    # aggregate cands over segments
    if not os.path.exists(os.path.join(workdir, 'cands_' + fileroot + '_sc' + str(scan) + '.pkl')):
        logger.info('Aggregating cands over segments %s for fileroot %s, scan %d' % (str(candssegs), fileroot, scan))
        logger.debug('%s' % candslist)

        cands = {}
        for candsfile in candslist:
            with open(candsfile, 'r') as pkl:
                state = pickle.load(pkl)
                result = pickle.load(pkl)
            for kk in result.keys():
                cands[kk] = result[kk]
            segment = state.pop('segment')  # remove this key, as it has no meaning after merging segments

        # optionally limit size
        if sizelimit and len(cands):
            logger.debug('Checking size of cands dictionary...')
            if 'snr2' in state['features']:
                snrcol = state['features'].index('snr2')
            elif 'snr1' in state['features']:
                snrcol = state['features'].index('snr1')

            candsize = sys.getsizeof(cands[cands.keys()[0]])/1e6
            maxlen = int(sizelimit/candsize)
            if len(cands) > maxlen:  # need to reduce length to newlen
                logger.info('cands dictionary of length %.1f would exceed sizelimit of %d MB. Trimming to strongest %d candidates' % (len(cands), sizelimit, maxlen))
                snrs = [abs(cands[k][snrcol]) for k in cands.iterkeys()]  # take top snrs
                snrsort = sorted(snrs, reverse=True)
                snrmax = snrsort[maxlen]  # get min snr for given length limit
                cands = {k: v for k,v in cands.items() if abs(v[snrcol]) > snrmax} # new cands dict

        # write cands to single file
        with open(os.path.join(workdir, 'cands_' + fileroot + '_sc' + str(scan) + '.pkl'), 'w') as pkl:
            pickle.dump(state, pkl, protocol=2)
            pickle.dump( (np.array(cands.keys()), np.array(cands.values())), pkl, protocol=2)
            
        if cleanup:
            if os.path.exists(os.path.join(workdir, 'cands_' + fileroot + '_sc' + str(scan) + '.pkl')):
                for candsfile in candslist:
                    os.remove(candsfile)
    else:
        logger.warn('Merged candsfile already exists for scan %d. Not merged.' % scan)

    # aggregate noise over segments
    if not os.path.exists(os.path.join(workdir, 'noise_' + fileroot + '_sc' + str(scan) + '.pkl')):
        logger.info('Aggregating noise over segments %s for fileroot %s, scan %d' % (str(noisesegs), fileroot, scan))
        logger.debug('%s' % noiselist)

        noise = []
        for noisefile in noiselist:
            with open(noisefile, 'r') as pkl:
                result = pickle.load(pkl)   # gets all noises for segment as list
            noise += result

        # write noise to single file
        if len(noise):
            with open(os.path.join(workdir, 'noise_' + fileroot + '_sc' + str(scan) + '.pkl'), 'w') as pkl:
                pickle.dump(noise, pkl, protocol=2)

        if cleanup:
            if os.path.exists(os.path.join(workdir, 'noise_' + fileroot + '_sc' + str(scan) + '.pkl')):
                for noisefile in noiselist:
                    os.remove(noisefile)

    else:
        logger.warn('Merged noisefile already exists for scan %d. Not merged.' % scan)


def cleanup(workdir, fileroot, scans=[]):
    """ Cleanup up noise and cands files.
    Finds all segments in each scan and merges them into single cand/noise file per scan.
    """

    os.chdir(workdir)

    # merge cands/noise files per scan
    for scan in scans:
        pc.merge_segments(fileroot, scan, cleanup=True, sizelimit=2.)


def merge_noises(pkllist, outroot=''):
    """ Merge noise files from multiple segments.
    Output noise file has scan number at start of each entry.
    """

    assert isinstance(pkllist, list), "pkllist must be list of file names"
    if not outroot:
        outroot = '_'.join(pkllist[0].split('_')[1:-1])

    workdir = os.path.dirname(pkllist[0])
    mergepkl = os.path.join(workdir, 'noise_' + outroot + '_merge.pkl')

    pkllist = [pkllist[i] for i in range(len(pkllist)) if ('merge' not in pkllist[i]) and ('seg' not in pkllist[i])]  # filter list down to per-scan noise pkls
    pkllist.sort(key=lambda i: int(i.rstrip('.pkl').split('_sc')[1]))  # sort by scan assuming filename structure
    scans = [int(ff.rstrip('.pkl').split('_sc')[1]) for ff in pkllist]
    logger.info('Aggregating noise from scans %s' % scans)

    allnoise = []
    for pklfile in pkllist:
        scan = int(pklfile.rstrip('.pkl').split('_sc')[1])   # parsing filename to get scan number
        with open(pklfile, 'r') as pkl:
            noises = pickle.load(pkl)   # gets all noises for segment as list
        allnoise += [[scan] + list(noise) for noise in noises]  # prepend scan number

    # write noise to single file
    if os.path.exists(mergepkl):
        logger.info('Overwriting merged noise file %s' % mergepkl)
        os.remove(mergepkl)
    else:
        logger.info('Writing merged noise file %s' % mergepkl)

    with open(mergepkl, 'w') as pkl:
        pickle.dump(allnoise, pkl, protocol=2)


def merge_cands(pkllist, outroot='', remove=[], snrmin=0, snrmax=999):
    """ Takes cands pkls from list and filteres to write new single "merge" pkl.
    Ignores segment cand files.
    remove is a list [t0,t1,t2,t3], where t0-t1, t2-t3 define the time ranges in seconds.
    snrmin, snrmax define how to filter cands read and written by abs(snr)
    """

    assert isinstance(pkllist, list), "pkllist must be list of file names"
    if not outroot:
        outroot = '_'.join(pkllist[0].split('_')[1:-1])

    workdir = os.path.dirname(pkllist[0])
    mergepkl = os.path.join(workdir, 'cands_' + outroot + '_merge.pkl')

    pkllist = [pkllist[i] for i in range(len(pkllist)) if ('merge' not in pkllist[i]) and ('seg' not in pkllist[i])]
    pkllist.sort(key=lambda i: int(i.rstrip('.pkl').split('_sc')[1]))  # assumes filename structure
    scans = [int(ff.rstrip('.pkl').split('_sc')[1]) for ff in pkllist]
    logger.info('Aggregating cands from scans %s' % scans)

    # get sample state dict. use 'dict' suffix to define multi-scan metadata dictionaries
    mergeloc = []; mergeprop = []; mergetimes = []
    segmenttimesdict = {}
    starttime_mjddict = {}
    for pklfile in pkllist:

        # get scan number and read candidates
        locs, props, d = read_candidates(pklfile, snrmin=snrmin, snrmax=snrmax, returnstate=True)
        if 'snr2' in d['features']:
            snrcol = d['features'].index('snr2')
        elif 'snr1' in d['features']:
            snrcol = d['features'].index('snr1')
        scan = int(pklfile.rstrip('.pkl').split('_sc')[1])   # parsing filename to get scan number
        segmenttimesdict[scan] = d['segmenttimes']
        starttime_mjddict[scan] = d['starttime_mjd']

        times = int2mjd(d, locs)

        # build merged loc,prop lists
        for i in range(len(locs)):
            loc = list(locs[i])
            loc.insert(0, scan)
            prop = list(props[i])
            mergeloc += [loc]
            mergeprop += [prop]
            mergetimes.append(times[i])

    mergeloc = np.array(mergeloc)
    mergeprop = np.array(mergeprop)
    mergetimes = np.array(mergetimes)

    # filter by remove, if needed
    if remove:
        mergetimes -= mergetimes.min()

        ww = np.ones(len(mergetimes), dtype=bool)  # initialize pass filter
        nranges = len(remove)
        for first in range(0,nranges,2):
            badrange0 = remove[first]
            badrange1 = remove[first+1]

            ww = ww & np.where( (mergetimes < badrange0) | (mergetimes > badrange1), True, False )

        mergeloc = mergeloc[ww]
        mergeprop = mergeprop[ww]

    # update metadata
    d['featureind'].insert(0, 'scan')
    d['remove'] = remove
    d['segmenttimesdict'] = segmenttimesdict
    d['starttime_mjddict'] = starttime_mjddict
    logger.info('Writing filtered set of %d candidates to %s' % (len(mergeloc), mergepkl))

    # write up new pkl
    pkl = open(mergepkl, 'w')
    pickle.dump(d, pkl, protocol=2)
    pickle.dump((mergeloc, mergeprop), pkl, protocol=2)
    pkl.close()


def merge_scans(workdir, fileroot, scans, snrmin=0, snrmax=999):
    """ Merge cands/noise files over all scans """


    pkllist = [ff for ff in
               [os.path.join(workdir, 'cands_{0}_sc{1}.pkl'.format(fileroot, scan))
                for scan in scans] if os.path.exists(ff)]
    merge_cands(pkllist, outroot=fileroot, snrmin=snrmin, snrmax=snrmax)

    pkllist = [ff for ff in
               [os.path.join(workdir, 'noise_{0}_sc{1}.pkl'.format(fileroot, scan))
                for scan in scans] if os.path.exists(ff)]
    merge_noises(pkllist, fileroot)


def split_candidates(candsfile, featind1, featind2, candsfile1, candsfile2):
    """ Split features from one candsfile into two new candsfiles

    featind1/2 is list of indices to take from d['features'].
    New features and updated state dict go to candsfile1/2.
    """

    with open(candsfile, 'rb') as pkl:
        d = pickle.load(pkl)
        cands = pickle.load(pkl)

    features = d['features']
    d1 = d.copy()
    d2 = d.copy()
    d1['features'] = [features[i] for i in featind1]
    d2['features'] = [features[i] for i in featind2]

    cands1 = {}
    cands2 = {}
    for key in cands:
        cands1[key] = tuple([cands[key][i] for i in featind1])
        cands2[key] = tuple([cands[key][i] for i in featind2])

    with open(candsfile1, 'w') as pkl:
        pickle.dump(d1, pkl, protocol=2)
        pickle.dump(cands1, pkl, protocol=2)

    with open(candsfile2, 'w') as pkl:
        pickle.dump(d2, pkl, protocol=2)
        pickle.dump(cands2, pkl, protocol=2)


def nbcompile(workdir, fileroot, html=True, basenb='', agdir=''):
    """ Run analysis pipeline from jupyter base notebook and save as notebook and html.

    html will also compile static html version
    basenb can be provided, else will get distributed version.
    agdir is the activegit repo (optional)
    """

    import inspect, rtpipe, shutil
    from subprocess import call

    os.environ['fileroot'] = fileroot
    if agdir:
        os.environ['agdir'] = agdir

    if not basenb:
        basenb = os.path.join(os.path.dirname(os.path.dirname(inspect.getfile(rtpipe))), 'notebooks/baseinteract.ipynb')

    logger.info('Moving to {0} and building notebook for {1}'.format(workdir, fileroot))
    os.chdir(workdir)
    shutil.copy(basenb, '{0}/{1}.ipynb'.format(workdir, fileroot))

    cmd = 'jupyter nbconvert {0}.ipynb --inplace --execute --to notebook --allow-errors --ExecutePreprocessor.timeout=3600'.format(fileroot).split(' ')
    status = call(cmd)

    cmd = 'jupyter trust {0}.ipynb'.format(fileroot).split(' ')
    status = call(cmd)

    if html:
        cmd = 'jupyter nbconvert {0}.ipynb --to html --output {0}.html'.format(fileroot).split(' ')
        status = call(cmd)


def thresholdcands(candsfile, threshold, numberperscan=1):
    """ Returns list of significant candidate loc in candsfile.
    Can define threshold and maximum number of locs per scan.
    Works on merge or per-scan cands pkls.
    """

    # read metadata and define columns of interest
    d = pickle.load(open(candsfile, 'r'))
    try:
        scancol = d['featureind'].index('scan')
    except ValueError:
        scancol = -1
    if 'snr2' in d['features']:
        snrcol = d['features'].index('snr2')
    elif 'snr1' in d['features']:
        snrcol = d['features'].index('snr1')

    # read data and define snrs
    loc, prop = pc.read_candidates(candsfile)
    snrs = [prop[i][snrcol] for i in range(len(prop)) if prop[i][snrcol] > threshold]

    # calculate unique list of locs of interest
    siglocs = [list(loc[i]) for i in range(len(prop)) if prop[i][snrcol] > threshold]
    siglocssort = sorted(zip([list(ll) for ll in siglocs], snrs), key=lambda stuff: stuff[1], reverse=True)

    if scancol >= 0:
        scanset = list(set([siglocs[i][scancol] for i in range(len(siglocs))]))
        candlist= []
        for scan in scanset:
            logger.debug('looking in scan %d' % scan)
            count = 0
            for sigloc in siglocssort:
                if sigloc[0][scancol] == scan:
                    logger.debug('adding sigloc %s' % str(sigloc))
                    candlist.append(sigloc)
                    count += 1
                if count >= numberperscan:
                    break
    else:
        candlist = siglocssort[:numberperscan]

    logger.debug('Returning %d cands above threshold %.1f' % (len(candlist), threshold))
    return [loc for loc,snr in candlist]


def postcands(mergepkl, url='http://localhost:9200/realfast/cands/_bulk?', snrmin=0, snrmax=999):
    """ Posts candidate info to elasticsearch index """

    loc, prop, d = read_candidates(mergepkl, snrmin=snrmin, snrmax=snrmax, returnstate=True)
    times = int2mjd(d, loc)

    alldata = []
    for i in range(len(loc)):
        data = {}
        data['filename'] = os.path.basename(d['filename'])
        data['@timestamp'] = times[i]
        for featureind in d['featureind']:
            data[featureind] = loc[i][d['featureind'].index(featureind)]
        for feature in d['features']:
            data[feature] = prop[i, d['features'].index(feature)]

        idobj = {}
        idobj['_id'] = '{:.7f}_{}_{}'.format(data['@timestamp'], data['dmind'], data['dtind'])
        
        alldata.append({"index":idobj})
        alldata.append(data)

    jsonStr = json.dumps(alldata, separators=(',', ':'))
    cleanjson = jsonStr.replace('}}, ','}}\n').replace('},', '}\n').replace(']', '').replace('[', '')
    r = requests.post(url, data=cleanjson)
    logger.debug('%s' % r)


def int2mjd(d, loc):
    """ Function to convert segment+integration into mjd seconds.
    """

    # needs to take merge pkl dict

    if len(loc):
        intcol = d['featureind'].index('int')
        segmentcol = d['featureind'].index('segment')
        if d.has_key('segmenttimesdict'):  # using merged pkl
            scancol = d['featureind'].index('scan')
            t0 = np.array([d['segmenttimesdict'][loc[i,scancol]][loc[i,segmentcol],0] for i in range(len(loc))])
        else:
            t0 = d['segmenttimes'][loc[:,segmentcol]][:,0]
        return (t0 + (d['inttime']/(24*3600.))*loc[:,intcol]) * 24*3600
    else:
        return np.array([])


def plot_full(candsfile, cands, mode='im'):
    """ Plot 'full' features, such as cutout image and spectrum.
    """

    loc, prop, d = read_candidates(candsfile, returnstate=True)
    npixx, npixy = prop[0][4].shape
    nints, nchan, npol = prop[0][5].shape

    bin = 10
    plt.figure(1)
    for i in cands:
        if mode == 'spec':
            rr = np.array([np.abs(prop[i][5][:,i0:i0+bin,0].mean(axis=1)) for i0 in range(0,nchan,bin)])
            ll = np.array([np.abs(prop[i][5][:,i0:i0+bin,1].mean(axis=1)) for i0 in range(0,nchan,bin)])
            sh = ll.shape
            data = np.concatenate( (rr, np.zeros(shape=(sh[0], sh[1]/2)), ll), axis=1)
        elif mode == 'im':
            data = prop[i][4]
        plt.subplot(np.sqrt(len(cands)), np.sqrt(len(cands)), cands.index(i))
        plt.imshow(data, interpolation='nearest')
    plt.show()


def make_psrrates(pkllist, nbins=60, period=0.156):
    """ Visualize cands in set of pkl files from pulsar observations.
    Input pkl list assumed to start with on-axis pulsar scan, followed by off-axis scans.
    nbins for output histogram. period is pulsar period in seconds (used to find single peak for cluster of detections).
    """

    # get metadata
    state = pickle.load(open(pkllist[0], 'r'))  # assume single state for all scans
    if 'image2' in state['searchtype']:
        immaxcol = state['features'].index('immax2')
        logger.info('Using immax2 for flux.')
    elif 'image1' in state['searchtype']:
        try:
            immaxcol = state['features'].index('immax1')
            logger.info('Using immax1 for flux.')
        except:
            immaxcol = state['features'].index('snr1')
            logger.info('Warning: Using snr1 for flux.')

    # read cands
    for pklfile in pkllist:
        loc, prop = read_candidates(pklfile)

        ffm = []
        if (loc):
            times = int2mjd(state, loc)

            for (mint,maxt) in zip(np.arange(times.min()-period/2,times.max()+period/2,period), np.arange(times.min()+period/2,times.max()+3*period/2,period)):
                ff = np.array([prop[i][immaxcol] for i in range(len(prop))])
                mm = ff[np.where( (times >= mint) & (times < maxt) )]
                if mm:
                    ffm.append(mm.max())
            ffm.sort()

        logger.info('Found %d unique pulses.' % len(ffm))
        # calculate params
        if pkllist.index(pklfile) == 0:
            duration0 = times.max() - times.min()
            ratemin = 1/duration0
            ratemax = len(ffm)/duration0
            rates = np.linspace(ratemin, ratemax, nbins)
            f0m = ffm
        elif pkllist.index(pklfile) == 1:
            duration1 = times.max() - times.min()
            f1m = ffm
        elif pkllist.index(pklfile) == 2:
            f2m = ffm
        elif pkllist.index(pklfile) == 3:
            f3m = ffm

    # calc rates
    f0 = []; f1 = []; f2 = []; f3 = []
    for rr in rates:
        num0 = (np.round(rr*duration0)).astype(int)
        num1 = (np.round(rr*duration1)).astype(int)
        
        if (num0 > 0) and (num0 <= len(f0m)):
            f0.append((rr,f0m[-num0]))

        if (num1 > 0) and (num1 <= len(f1m)):
            f1.append((rr,f1m[-num1]))

        if (num1 > 0) and (num1 <= len(f2m)):
            f2.append((rr,f2m[-num1]))

        if len(pkllist) == 4:
            if f3m:
                if (num1 > 0) and (num1 <= len(f3m)):
                    f3.append((rr,f3m[-num1]))

    if f3:
        return {0: np.array(f0).transpose(), 1: np.array(f1).transpose(), 2: np.array(f2).transpose(), 3: np.array(f3).transpose()}
    else:
        return {0: np.array(f0).transpose(), 1: np.array(f1).transpose(), 2: np.array(f2).transpose()}


def plot_psrrates(pkllist, outname=''):
    """ Plot cumulative rate histograms. List of pkl files in order, as for make_psrrates.
    """

    if not outname:
        outname = 'tmp.png'

    labels = {0: 'Flux at 0\'', 1: 'Flux at 7\'', 2: 'Flux at 15\'', 3: 'Flux at 25\''}
    labelsr = {1: 'Flux Ratio 7\' to 0\'', 2: 'Flux Ratio 15\' to 0\'', 3: 'Flux Ratio 25\' to 0\''}
    colors = {0: 'b.', 1: 'r.', 2: 'g.', 3: 'y.'}

    rates = make_psrrates(pkllist)
    plt.clf()
    fig = plt.figure(1, figsize=(10,8), facecolor='white')
    ax = fig.add_subplot(211, axis_bgcolor='white')
    for kk in rates.keys():
        flux, rate = rates[kk]
        plt.plot(flux, rate, colors[kk], label=labels[kk])

    plt.setp( ax.get_xticklabels(), visible=False)
    plt.ylabel('Flux (Jy)', fontsize='20')
    plt.legend(numpoints=1)
    plt.loglog()

    ax2 = fig.add_subplot(212, sharex=ax, axis_bgcolor='white')
    flux0, rate0 = rates[0]
    for kk in rates.keys():
        flux, rate = rates[kk]
        if kk == 1:
            r10 = [rate[i]/rate0[np.where(flux0 == flux[i])[0][0]] for i in range(len(rate))]
            plt.plot(flux, r10, colors[kk], label=labelsr[kk])
        elif kk == 2:
            r20 = [rate[i]/rate0[np.where(flux0 == flux[i])[0][0]] for i in range(len(rate))]
            plt.plot(flux, r20, colors[kk], label=labelsr[kk])
        elif kk == 3:
            r30 = [rate[i]/rate0[np.where(flux0 == flux[i])[0][0]] for i in range(len(rate))]
            plt.plot(flux, r30, colors[kk], label=labelsr[kk])

    plt.xlabel('Rate (1/s)', fontsize='20')
    plt.ylabel('Flux ratio', fontsize='20')
    plt.legend(numpoints=1)
    plt.subplots_adjust(hspace=0)

    # find typical ratio. avoid pulsar period saturation and low-count regimes (high and low ends)
    if len(rates) == 4:
        logger.info('flux ratio, lowest common (1/0, 2/0, 3/0):', (r10[len(r30)-1], r20[len(r30)-1], r30[-1]))
        logger.info('flux ratio, high end (1/0, 2/0, 3/0):', (r10[-1], r20[-1], r30[-1]))
    elif len(rates) == 3:
        logger.info('flux ratio, lowest common (1/0, 2/0):', (r10[len(r20)-1], r20[-1]))
        logger.info('flux ratio, high end (1/0, 2/0):', (r10[-1], r20[-1]))

    plt.savefig(outname)


def mock_fluxratio(candsfile, mockcandsfile, dmbin=0):
    """ Associates mock cands with detections in candsfile by integration.
    Returns ratio of detected to expected flux for all associations.
    """

    loc, prop = read_candidates(candsfile)
    loc2, prop2 = read_candidates(mockcandsfile)

    dmselect = np.where(loc[:,2] == dmbin)[0]
    mocki = [i for i in loc2[:,1].astype(int)]  # known transients
    rat = []; newloc = []; newprop = []
    for i in mocki:
        try:
            detind = list(loc[dmselect,1]).index(i)   # try to find detection
            rat.append(prop[dmselect][detind][1]/prop2[mocki.index(i)][1])
            newloc.append(list(loc2[mocki.index(i)]))
            newprop.append(list(prop2[mocki.index(i)]))
        except ValueError:
            pass

    return rat, np.array(newloc), newprop
