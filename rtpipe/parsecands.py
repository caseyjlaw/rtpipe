import numpy as n
from scipy.special import erfinv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pickle, types, glob, os
import rtpipe.RT as rt

def read_candidates(candsfile):
    """ Reads candidate pkl file into numpy array.
    Returns tuple of two numpy arrays (location, features).
    """

    # read in pickle file of candidates
    with open(candsfile, 'rb') as pkl:
        d = pickle.load(pkl)
        cands = pickle.load(pkl)
    if cands == 0:
        print 'No cands found from %s.' % candsfile
        return (n.array([]), n.array([]))

    # select set of values 
    loc = []; prop = []
    for kk in sorted(cands.keys()):
        loc.append( list(kk) )
        prop.append( list(cands[kk]) )    #[snrcol], cands[kk][l1col], cands[kk][m1col]) )

    print 'Read %d candidates from %s.' % (len(loc), candsfile)
    return n.array(loc).astype(int), n.array(prop)

def read_noise(noisefile):
    """ Function to read a noise file and parse columns.
    """

    f = open(noisefile,'r')
    noises = pickle.load(f)
    seg = []; noiseperbl = []; flagfrac = []; imnoise = []
    for noise in noises:
        seg.append(noise[0]); noiseperbl.append(noise[1])
        flagfrac.append(noise[2]); imnoise.append(noise[3])
    return (n.array(seg), n.array(noiseperbl), n.array(flagfrac), n.array(imnoise))

def merge_segments(pkllist, fileroot=''):
    """ Merges cands/noise pkl files from multiple segments to single cands/noise file.
    Output single pkl per scan with root name fileroot.
    """

    assert pkllist > 0, 'pkllist is empty'

    workdir = os.path.dirname(pkllist[0])

    if not fileroot:
        fileroot = '_'.join(pkllist[0].split('seg')[0].split('_')[1:])   # assumes filename structure

    # aggregate cands over segments
    if 'cands' in pkllist[0]:
        print 'Aggregating cands from %s' % pkllist
        state = pickle.load(open(pkllist[0], 'r'))
        cands = {}
        for cc in pkllist:
            with open(cc,'r') as pkl:
                state = pickle.load(pkl)
                result = pickle.load(pkl)
            for kk in result.keys():
                cands[kk] = result[kk]

        # write cands to single file
        with open(os.path.join(workdir, 'cands_' + fileroot + '.pkl'), 'w') as pkl:
            pickle.dump(state, pkl)
            pickle.dump(cands, pkl)

    # clean up noise files
    elif 'noise' in pkllist[0]:
        print 'Aggregating noise from %s' % pkllist
        # aggregate noise over segments
        noise = []
        for cc in pkllist:
            try:
                with open(cc,'r') as pkl:
                    result = pickle.load(pkl)
                noise += result
            except EOFError:
                pass

        # write noise to single file
        with open(os.path.join(workdir, 'noise_' + fileroot + '.pkl'), 'w') as pkl:
            pickle.dump(noise, pkl)

    else:
        print 'Don\'t know what to do with pkllist: %s' % (str(pkllist))

def merge_cands(pkllist, outroot='', remove=[]):
    """ Takes cands pkls from list and filteres to write new single "merge" pkl.
    Ignores segment cand files.
    remove is a list [t0,t1,t2,t3], where t0-t1, t2-t3 define the time ranges in seconds.
    """

    assert isinstance(pkllist, list), "pkllist must be list of file names"
    if not outroot:
        outroot = '_'.join(pkllist[0].split('_')[1:3])
    mergepkl = 'cands_' + outroot + '_merge.pkl'

    pkllist = [pkllist[i] for i in range(len(pkllist)) if ('merge' not in pkllist[i]) and ('seg' not in pkllist[i])]
    pkllist.sort(key=lambda i: int(i.rstrip('.pkl').split('_sc')[1]))  # assumes filename structure
    print 'Aggregating cands from %s' % pkllist

    # get sample state dict. not representative of all scans
    mergeloc = []; mergeprop = []; mergetimes = []
    segmenttimesdict = {}
    for pklfile in pkllist:

        # get scan number and read candidates
        d = pickle.load(open(pklfile, 'r'))
        scan = int(pklfile.rstrip('.pkl').split('_sc')[1])   # parsing filename to get scan number
        segmenttimesdict[scan] = d['segmenttimes']

        locs, props = read_candidates(pklfile)
        times = int2mjd(d, n.array(locs))

        # build merged loc,prop lists
        for i in range(len(locs)):
            loc = list(locs[i])
            loc.insert(0, scan)
            prop = list(props[i])
            mergeloc += [loc]
            mergeprop += [prop]
            mergetimes.append(times[i])

    mergeloc = n.array(mergeloc)
    mergeprop = n.array(mergeprop)
    mergetimes = n.array(mergetimes)

    # filter by remove, if needed
    if remove:
        mergetimes -= mergetimes.min()

        ww = n.ones(len(mergetimes), dtype=bool)  # initialize pass filter
        nranges = len(remove)
        for first in range(0,nranges,2):
            badrange0 = remove[first]
            badrange1 = remove[first+1]

            ww = ww & n.where( (mergetimes < badrange0) | (mergetimes > badrange1), True, False )

        mergeloc = mergeloc[ww]
        mergeprop = mergeprop[ww]

# **todo**
    # if the scan has any candidates, add nints to count
    # goodintcount = 0
    # for scani in n.unique(dataloc):
    #     if len(n.where(dataloc == scani)[0]):
    #         goodintcount += d['nints']

    # # correct int count by removed range
    # for scan in remove.keys():
    #     goodintcount -= remove[scan][1] - remove[scan][0]

    # update metadata
#    d['goodintcount'] = goodintcount
    d['featureind'].insert(0, 'scan')
    d['remove'] = remove
    d['segmenttimesdict'] = segmenttimesdict

    print 'Writing filtered set of %d candidates to %s' % (len(mergeloc), mergepkl)

    # build and write up new dict
    cands = {}
    for i in range(len(mergeloc)):
        cands[tuple(mergeloc[i])] = tuple(mergeprop[i])

    pkl = open(os.path.join(d['workdir'], mergepkl), 'w')
    pickle.dump(d, pkl)
    pickle.dump(cands, pkl)
    pkl.close()

def plot_summary(pkllist, outroot='', remove=[]):
    """ Take pkl list or merge file to produce comprehensive candidate screening plots.
    Starts as dm-t plots, includes dt and peak pixel location.
    """

    # if a list, merge them
    if isinstance(pkllist, list):
        workdir = os.path.dirname(pkllist[0])

        if not outroot:
            outroot = '_'.join(pkllist[0].split('_')[1:3])
        merge_cands(pkllist, outroot=outroot, remove=remove)
        mergepkl = os.path.join(workdir, 'cands_' + outroot + '_merge.pkl')
    elif isinstance(pkllist, str):
        print 'Assuming input is mergepkl. Not using remove!'
        if not outroot:
            outroot = '_'.join(pkllist.split('_')[1:3])
        mergepkl = pkllist
    else:
        print 'Not valid input.'

    d = pickle.load(open(mergepkl, 'r'))
    locs, props = read_candidates(mergepkl)

    if not len(locs):
        print 'No candidates in mergepkl.'
        return

    # compile candidates over all pkls
    # feature columns
    if 'snr1' in d['features']:
        snrcol = d['features'].index('snr1')
    if 'l1' in d['features']:
        l1col = d['features'].index('l1')
    if 'm1' in d['features']:
        m1col = d['features'].index('m1')
        
    dtindcol = d['featureind'].index('dtind')
    dmindcol = d['featureind'].index('dmind')

    # extract values for plotting
    times = int2mjd(d, locs)
    times -= times.min()
    dts = locs[:, dtindcol]
    dms = n.array(d['dmarr'])[locs[:,dmindcol]]
    snrs = props[:, snrcol]
    l1s = props[:, l1col]
    m1s = props[:, m1col]

    # dmt plot
    print 'Plotting DM-time distribution...'
    plot_dmt(d, times, dms, dts, snrs, outroot)

    # dmcount plot
    print 'Plotting DM count distribution...'
    plot_dmcount(d, times, dts, outroot)

    # norm prob plot
    print 'Plotting normal probability distribution...'
    plot_normprob(d, snrs, outroot)

    # source location plot
    print 'Plotting (l,m) distribution...'
    plot_lm(d, snrs, l1s, m1s, outroot)

def plot_noise(pkllist, outroot='', remove=[]):
    """ Takes merged noise pkl and visualizes it.
    """

    if not outroot:
        outroot = '_'.join(pkllist[0].split('_')[1:3])

    make_noisehists(pkllist, outroot, remove=remove)

def int2mjd(d, loc):
    """ Function to convert segment+integration into mjd seconds.
    """

    # needs to take merge pkl dict

    if loc:
        intcol = d['featureind'].index('int')
        segmentcol = d['featureind'].index('segment')
        if d.has_key('segmenttimesdict'):  # using merged pkl
            scancol = d['featureind'].index('scan')
            t0 = n.array([d['segmenttimesdict'][loc[i,scancol]][loc[i,segmentcol],0] for i in range(len(loc))])
        else:
            t0 = d['segmenttimes'][loc[:,segmentcol]][:,0]
        return (t0 + (d['inttime']/(24*3600.))*loc[:,intcol]) * 24*3600
    else:
        return n.array([])

def plot_dmt(d, times, dms, dts, snrs, outroot):
    """ Plots DM versus time for each dt value.
    """

    outname = os.path.join(d['workdir'], 'plot_' + outroot + '_dmt.png')

    mint = times.min(); maxt = times.max()
    dtsunique = n.unique(dts)
    mindm = min(d['dmarr']); maxdm = max(d['dmarr'])
    snrmin = 0.8*min(d['sigma_image1'], d['sigma_image2'])

    fig = plt.Figure(figsize=(15,10))
    ax = {}
    for dtind in range(len(dtsunique)):
        ax[dtind] = fig.add_subplot(str(len(dtsunique)) + '1' + str(dtind+1))
        # plot positive cands
        good = n.where( (dts == dtind) & (snrs > 0))[0]
        sizes = (snrs[good]-snrmin)**5   # set scaling to give nice visual sense of SNR
        ax[dtind].scatter(times[good], dms[good], s=sizes, facecolor='none', alpha=0.3, clip_on=False)
        # plot positive cands
        good = n.where( (dts == dtind) & (snrs < 0))[0]
        sizes = (n.abs(snrs[good])-snrmin)**5   # set scaling to give nice visual sense of SNR
        ax[dtind].scatter(times[good], dms[good], s=sizes, marker='x', edgecolors='k', alpha=0.3, clip_on=False)

        ax[dtind].axis( (mint, maxt, mindm, maxdm) )
        ax[dtind].set_ylabel('DM (pc/cm3)')
        ax[dtind].text(0.9*maxt, 0.9*maxdm, 'dt='+str(dtsunique[dtind]))
        if dtind == dtsunique[-1]:
            plt.setp(ax[dtind].get_xticklabels(), visible=True)
        elif dtind == dtsunique[0]:
            ax[dtind].xaxis.set_label_position('top')
            ax[dtind].xaxis.set_ticks_position('top')
#            ax[dtind].set_xticks(changepoints[::2]*d['inttime']*d['nints'])
#            ax[dtind].set_xticklabels(changepoints[::2])
            plt.setp( ax[dtind].xaxis.get_majorticklabels(), rotation=90)
        else:
            plt.setp( ax[dtind].get_xticklabels(), visible=False)

    ax[dtind].set_xlabel('Time (s)', fontsize=20)
#    ax[dtind].set_xlabel('Scan number', fontsize=20)
    ax[dtind].set_ylabel('DM (pc/cm3)', fontsize=20) 
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(outname)

def plot_dmcount(d, times, dts, outroot):
    """ Count number of candidates per dm and dt. Big clusters often trace RFI.
    """

    outname = os.path.join(d['workdir'], 'plot_' + outroot + '_dmcount.png')

    uniquedts = n.unique(dts)
    mint = times.min(); maxt = times.max()

    fig2 = plt.Figure(figsize=(15,10))
    ax2 = {}
    for dtind in range(len(uniquedts)):
        good = n.where(dts == dtind)[0]
        ax2[dtind] = fig2.add_subplot(str(len(uniquedts)) + '1' + str(dtind+1))
        if good:
            bins = n.round(times[good]).astype('int')
            counts = n.bincount(bins)

            ax2[dtind].scatter(n.arange(n.amax(bins)+1), counts, facecolor='none', alpha=0.5, clip_on=False)
            ax2[dtind].axis( (mint, maxt, 0, 1.1*counts.max()) )

            # label high points
            high = n.where(counts > n.median(counts) + 20*counts.std())[0]
            print
            print 'Candidate clusters for dt=%d:' % (d['dtarr'][dtind])
            print '\t(Counts, Times)'
            for ii in high:
#                print 'For dt=%d, %d candidates at %d s' % (d['dtarr'][dtind], counts[ii], ii)
                ww = n.where(bins == ii)[0]
                print counts[ii], times[good][ww][0]

            if dtind == uniquedts[-1]:
                plt.setp(ax2[dtind].get_xticklabels(), visible=True)
            elif (dtind == uniquedts[0]) or (dtind == len(uniquedts)/2):
                ax2[dtind].xaxis.set_label_position('top')
                ax2[dtind].xaxis.set_ticks_position('top')
#            ax2[dtind].set_xticks(changepoints[::2]*d['nints']*d['inttime'])
#            ax2[dtind].set_xticklabels(changepoints[::2])
                plt.setp( ax2[dtind].xaxis.get_majorticklabels(), rotation=90, size='small')
            else:
                plt.setp( ax2[dtind].get_xticklabels(), visible=False)

    ax2[dtind].set_xlabel('Time (s)')
    ax2[dtind].set_ylabel('Count') 
    canvas2 = FigureCanvasAgg(fig2)
    canvas2.print_figure(outname)

def plot_normprob(d, snrs, outroot):
    """ Normal quantile plot compares observed SNR to expectation given frequency of occurrence.
    Includes negative SNRs, too.
    """

    outname = os.path.join(d['workdir'], 'plot_' + outroot + '_normprob.png')

    # define norm quantile functions
    Z = lambda quan: n.sqrt(2)*erfinv( 2*quan - 1) 
    quan = lambda ntrials, i: (ntrials + 1/2. - i)/ntrials

    # calc number of trials
    npix = d['npixx']*d['npixy']
    if d.has_key('goodintcount'):
        nints = d['goodintcount']
    else:
        nints = d['nints']
    ndms = len(d['dmarr'])
    dtfactor = n.sum([1./i for i in d['dtarr']])    # assumes dedisperse-all algorithm
    ntrials = npix*nints*ndms*dtfactor

    # calc normal quantile
    if len(n.where(snrs > 0)[0]):
        snrsortpos = n.array(sorted(snrs[n.where(snrs > 0)], reverse=True))     # high-res snr
        Zsortpos = n.array([Z(quan(ntrials, j+1)) for j in range(len(snrsortpos))])
    if len(n.where(snrs < 0)[0]):
        snrsortneg = n.array(sorted(n.abs(snrs[n.where(snrs < 0)]), reverse=True))     # high-res snr
        Zsortneg = n.array([Z(quan(ntrials, j+1)) for j in range(len(snrsortneg))])

    # plot
    fig3 = plt.Figure(figsize=(10,10))
    ax3 = fig3.add_subplot(111)
    if len(n.where(snrs < 0)[0]) and len(n.where(snrs > 0)[0]):
        ax3.plot(snrsortpos, Zsortpos, 'k.')
        ax3.plot(snrsortneg, Zsortneg, 'kx')
        refl = n.linspace(min(snrsortpos.min(), Zsortpos.min(), snrsortneg.min(), Zsortneg.min()), max(snrsortpos.max(), Zsortpos.max(), snrsortneg.max(), Zsortneg.max()), 2)
    elif len(n.where(snrs > 0)[0]):
        refl = n.linspace(min(snrsortpos.min(), Zsortpos.min()), max(snrsortpos.max(), Zsortpos.max()), 2)
    elif len(n.where(snrs < 0)[0]):
        refl = n.linspace(min(snrsortneg.min(), Zsortneg.min()), max(snrsortneg.max(), Zsortneg.max()), 2)
    ax3.plot(refl, refl, 'k--')
    ax3.set_xlabel('SNR')
    ax3.set_ylabel('Normal quantile SNR')
    canvas = FigureCanvasAgg(fig3)
    canvas.print_figure(outname)

def plot_lm(d, snrs, l1s, m1s, outroot):
    """ Plot the lm coordinates (relative to phase center) for all candidates.
    """

    outname = os.path.join(d['workdir'], 'plot_' + outroot + '_impeak.png')

    snrmin = 0.8*min(d['sigma_image1'], d['sigma_image2'])
    fig4 = plt.Figure(figsize=(10,10))
    ax4 = fig4.add_subplot(111)

    # plot positive
    good = n.where(snrs > 0)
    sizes = (snrs[good]-snrmin)**5   # set scaling to give nice visual sense of SNR
    xarr = 60*n.degrees(l1s[good]); yarr = 60*n.degrees(m1s[good])
    ax4.scatter(xarr, yarr, s=sizes, facecolor='none', alpha=0.5, clip_on=False)
    # plot negative
    good = n.where(snrs < 0)
    sizes = (n.abs(snrs[good])-snrmin)**5   # set scaling to give nice visual sense of SNR
    xarr = 60*n.degrees(l1s[good]); yarr = 60*n.degrees(m1s[good])
    ax4.scatter(xarr, yarr, s=sizes, marker='x', edgecolors='k', alpha=0.5, clip_on=False)

    ax4.set_xlabel('Dec Offset (amin)')
    ax4.set_ylabel('RA Offset (amin)')
    fov = n.degrees(1./d['uvres'])*60.
    ax4.set_xlim(fov/2, -fov/2)
    ax4.set_ylim(-fov/2, fov/2)
    canvas4 = FigureCanvasAgg(fig4)
    canvas4.print_figure(outname)

def make_noisehists(pkllist, outroot, remove=[]):
    """ Cumulative hist of image noise levels.
    """

    assert len(pkllist) > 0
    workdir = os.path.dirname(pkllist[0])

    outname = os.path.join(workdir, 'plot_' + outroot + '_noisehist.png')

    noises = []; minnoise = 1e8; maxnoise = 0
    print 'Reading %d noise files' % len(pkllist)
    for pkl in pkllist:
        seg, noiseperbl, flagfrac, imnoise = read_noise(pkl)

        if remove: print 'Remove option not supported for noise files yet.'

        noises.append(imnoise)  # TBD: filter this by remove
        minnoise = min(minnoise, imnoise.min())
        maxnoise = max(maxnoise, imnoise.max())

    bins = n.linspace(minnoise, maxnoise, 50)
    fig = plt.Figure(figsize=(10,10))
    ax = fig.add_subplot(211, axisbg='white')
    stuff = ax.hist(noises, bins=bins, histtype='bar', lw='none', ec='none')
    ax.set_title('Histograms of noise samples')
    ax.set_xlabel('Image RMS (Jy)')
    ax.set_ylabel('Number of noise measurements')
    ax2 = fig.add_subplot(212, axisbg='white')
    stuff = ax2.hist(n.array([noises[i][j] for i in range(len(noises)) for j in range(len(noises[i]))]), bins=bins, cumulative=-1, normed=True, log=False, histtype='bar', lw='none', ec='none')
    ax2.set_xlabel('Image RMS (Jy)')
    ax2.set_ylabel('Number with noise > image RMS')

    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(outname)

def make_psrrates(pkllist, nbins=60, period=0.156):
    """ Visualize cands in set of pkl files from pulsar observations.
    Input pkl list assumed to start with on-axis pulsar scan, followed by off-axis scans.
    nbins for output histogram. period is pulsar period in seconds (used to find single peak for cluster of detections).
    """

    # get metadata
    state = pickle.load(open(pkllist[0], 'r'))  # assume single state for all scans
    if 'image2' in state['searchtype']:
        immaxcol = state['features'].index('immax2')
        print 'Using immax2 for flux.'
    elif 'image1' in state['searchtype']:
        try:
            immaxcol = state['features'].index('immax1')
            print 'Using immax1 for flux.'
        except:
            immaxcol = state['features'].index('snr1')
            print 'Warning: Using snr1 for flux.'

    # read cands
    for pklfile in pkllist:
        loc, prop = read_candidates(pklfile)

        ffm = []
        if loc:
            times = int2mjd(state, loc)

            for (mint,maxt) in zip(n.arange(times.min()-period/2,times.max()+period/2,period), n.arange(times.min()+period/2,times.max()+3*period/2,period)):
                ff = prop[:,immaxcol]
                mm = ff[n.where( (times >= mint) & (times < maxt) )]
                if mm:
                    ffm.append(mm.max())
            ffm.sort()

        print 'Found %d unique pulses.' % len(ffm)
        # calculate params
        if pkllist.index(pklfile) == 0:
            duration0 = times.max() - times.min()
            ratemin = 1/duration0
            ratemax = len(ffm)/duration0
            rates = n.linspace(ratemin, ratemax, nbins)
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
        num0 = (n.round(rr*duration0)).astype(int)
        num1 = (n.round(rr*duration1)).astype(int)
        
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
        return {0: n.array(f0).transpose(), 1: n.array(f1).transpose(), 2: n.array(f2).transpose(), 3: n.array(f3).transpose()}
    else:
        return {0: n.array(f0).transpose(), 1: n.array(f1).transpose(), 2: n.array(f2).transpose()}

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
            r10 = [rate[i]/rate0[n.where(flux0 == flux[i])[0][0]] for i in range(len(rate))]
            plt.plot(flux, r10, colors[kk], label=labelsr[kk])
        elif kk == 2:
            r20 = [rate[i]/rate0[n.where(flux0 == flux[i])[0][0]] for i in range(len(rate))]
            plt.plot(flux, r20, colors[kk], label=labelsr[kk])
        elif kk == 3:
            r30 = [rate[i]/rate0[n.where(flux0 == flux[i])[0][0]] for i in range(len(rate))]
            plt.plot(flux, r30, colors[kk], label=labelsr[kk])

    plt.xlabel('Rate (1/s)', fontsize='20')
    plt.ylabel('Flux ratio', fontsize='20')
    plt.legend(numpoints=1)
    plt.subplots_adjust(hspace=0)

    # find typical ratio. avoid pulsar period saturation and low-count regimes (high and low ends)
    if len(rates) == 4:
        print 'flux ratio, lowest common (1/0, 2/0, 3/0):', (r10[len(r30)-1], r20[len(r30)-1], r30[-1])
        print 'flux ratio, high end (1/0, 2/0, 3/0):', (r10[-1], r20[-1], r30[-1])
    elif len(rates) == 3:
        print 'flux ratio, lowest common (1/0, 2/0):', (r10[len(r20)-1], r20[-1])
        print 'flux ratio, high end (1/0, 2/0):', (r10[-1], r20[-1])

    plt.savefig(outname)

def plot_cand(mergepkl, snrmin=None, candnum=-1, outname='', **kwargs):
    """ Create detailed plot of a single candidate.
    Thresholds (as minimum), then provides list of candidates to select with candnum.
    kwargs passed to rt.set_pipeline
    """

    # if isinstance(pkllist, list):
    #     outroot = '_'.join(pkllist[0].split('_')[1:3])
    #     merge_cands(pkllist, outroot=outroot)
    #     mergepkl = 'cands_' + outroot + '_merge.pkl'
    # elif isinstance(pkllist, str):
    #     print 'Assuming input is mergepkl'
    #     mergepkl = pkllist

    d = pickle.load(open(mergepkl, 'r'))
    loc, prop = read_candidates(mergepkl)
    
    if not os.path.dirname(d['filename']):
        d['filename'] = os.path.join(d['workdir'], d['filename'])

    # feature columns
    if 'snr2' in d['features']:
        snrcol = d['features'].index('snr2')
    elif 'snr1' in d['features']:
        snrcol = d['features'].index('snr1')
    if 'l2' in d['features']:
        lcol = d['features'].index('l2')
    elif 'l1' in d['features']:
        lcol = d['features'].index('l1')
    if 'm2' in d['features']:
        mcol = d['features'].index('m2')
    elif 'm1' in d['features']:
        mcol = d['features'].index('m1')
        
    scancol = d['featureind'].index('scan')
    segmentcol = d['featureind'].index('segment')
    intcol = d['featureind'].index('int')
    dtindcol = d['featureind'].index('dtind')
    dmindcol = d['featureind'].index('dmind')

    # sort and prep candidate list
    snrs = prop[:,snrcol]
    if isinstance(snrmin, type(None)):
        snrmin = min(snrs)
    sortord = snrs.argsort()
    snrinds = n.where(snrs[sortord] > snrmin)[0]
    loc = loc[sortord][snrinds]
    prop = prop[sortord][snrinds]

    if candnum < 0:
        print 'Getting candidates...'
        for i in range(len(loc)):
            print i, loc[i], prop[i, snrcol]
        return (loc, prop[:,snrcol])
    else:
        print 'Reproducing and visualizing candidate %d at %s with properties %s.' % (candnum, loc[candnum], prop[candnum])
        scan = loc[candnum, scancol]
        segment = loc[candnum, segmentcol]
        dmind = loc[candnum, dmindcol]
        dtind = loc[candnum, dtindcol]
        candint = loc[candnum, intcol]
        dmarrorig = d['dmarr']
        dtarrorig = d['dtarr']
        nsegments = len(d['segmenttimesdict'][scan])

        d2 = rt.set_pipeline(d['filename'], scan, fileroot=d['fileroot'], paramfile='rtparams.py', savecands=False, savenoise=False, nsegments=nsegments, **kwargs)
        im, data = rt.pipeline_reproduce(d2, segment, (candint, dmind, dtind))  # with candnum, pipeline will return cand image and data

        # calc source location
        peakl, peakm = n.where(im == im.max())
        xpix,ypix = im.shape
        l1 = (xpix/2. - peakl[0])/(xpix*d['uvres'])
        m1 = (ypix/2. - peakm[0])/(ypix*d['uvres'])
        pt_ra, pt_dec = d['radec']
        src_ra, src_dec = source_location(pt_ra, pt_dec, l1, m1)
        print 'Peak (RA, Dec):', src_ra, src_dec

        # plot it
        print 'Plotting...'
        fig = plt.Figure(figsize=(8.5,8))
        ax = fig.add_subplot(221, axisbg='white')

        # # first plot dm-t distribution beneath
        times0 = int2mjd(d, loc)
        times0 = times0 - times0[0]
        times = times0[candnum]
        dms0 = n.array(dmarrorig)[list(loc[:,dmindcol])]
        dms = dmarrorig[loc[candnum,dmindcol]]
        snr0 = prop[:, snrcol]
        snr = prop[candnum, snrcol]
        snrmin = 0.8 * min(d['sigma_image1'], d['sigma_image2'])
        # plot positive
        good = n.where(snr0 > 0)
        ax.scatter(times0[good], dms0[good], s=(snr0[good]-snrmin)**5, facecolor='none', linewidth=0.2, clip_on=False)
        ax.scatter(times, dms, s=(snr-snrmin)**5, facecolor='none', linewidth=2, clip_on=False)
        # plot negative
        good = n.where(snr0 < 0)
        ax.scatter(times0[good], dms0[good], s=(n.abs(snr0)[good]-snrmin)**5, marker='x', edgecolors='k', linewidth=0.2, clip_on=False)
        ax.set_ylim(dmarrorig[0], dmarrorig[-1])
        ax.set_xlim(times0.min(), times0.max())
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('DM (pc/cm3)')

        # # then add annotating info
        ax.text(0.1, 0.9, d['fileroot']+'_sc'+str(scan), fontname='sans-serif', transform = ax.transAxes)
        ax.text(0.1, 0.8, 'seg %d, int %d, DM %.1f, dt %d' % (segment, loc[candnum, intcol], dmarrorig[loc[candnum, dmindcol]], dtarrorig[loc[candnum,dtindcol]]), fontname='sans-serif', transform = ax.transAxes)

        ax.text(0.1, 0.7, 'Peak: (' + str(n.round(peakx, 1)) + '\' ,' + str(n.round(peaky, 1)) + '\'), SNR: ' + str(n.round(snr, 1)), fontname='sans-serif', transform = ax.transAxes)

        # plot dynamic spectra
        left, width = 0.6, 0.2
        bottom, height = 0.2, 0.7
        rect_dynsp = [left, bottom, width, height]
        rect_lc = [left, bottom-0.1, width, 0.1]    
        rect_sp = [left+width, bottom, 0.1, height]
        ax_dynsp = fig.add_axes(rect_dynsp)
        ax_lc = fig.add_axes(rect_lc)    
        ax_sp = fig.add_axes(rect_sp)
        spectra = n.swapaxes(data.real,0,1)      # seems that latest pickle actually contains complex values in spectra...
        dd = n.concatenate( (spectra[...,0], n.zeros_like(spectra[...,0]), spectra[...,1]), axis=1)    # make array for display with white space between two pols
        impl = ax_dynsp.imshow(dd, origin='lower', interpolation='nearest', aspect='auto', cmap=plt.get_cmap('Greys'))
        ax_dynsp.text(0.5, 0.95, 'RR LL', horizontalalignment='center', verticalalignment='center', fontsize=16, color='w', transform = ax_dynsp.transAxes)
        ax_dynsp.set_yticks(range(0,len(d['freq']),30))
        ax_dynsp.set_yticklabels(d['freq'][::30])
        ax_dynsp.set_ylabel('Freq (GHz)')
        ax_dynsp.set_xlabel('Integration (rel)')
        spectrum = spectra[:,len(spectra[0])/2].mean(axis=1)      # assume pulse in middle bin. get stokes I spectrum. **this is wrong in a minority of cases.**
        ax_sp.plot(spectrum, range(len(spectrum)), 'k.')
        ax_sp.plot(n.zeros(len(spectrum)), range(len(spectrum)), 'k:')
        ax_sp.set_ylim(0, len(spectrum))
        ax_sp.set_yticklabels([])
        xmin,xmax = ax_sp.get_xlim()
        ax_sp.set_xticks(n.linspace(xmin,xmax,3).round(2))
        ax_sp.set_xlabel('Flux (Jy)')
        lc = dd.mean(axis=0)
        lenlc = n.where(lc == 0)[0][0]
        ax_lc.plot(range(0,lenlc)+range(2*lenlc,3*lenlc), list(lc)[:lenlc] + list(lc)[-lenlc:], 'k.')
        ax_lc.plot(range(0,lenlc)+range(2*lenlc,3*lenlc), list(n.zeros(lenlc)) + list(n.zeros(lenlc)), 'k:')
        ax_lc.set_xlabel('Integration')
        ax_lc.set_ylabel('Flux (Jy)')
        ax_lc.set_xticks([0,0.5*lenlc,lenlc,1.5*lenlc,2*lenlc,2.5*lenlc,3*lenlc])
        ax_lc.set_xticklabels(['0',str(lenlc/2),str(lenlc),'','0',str(lenlc/2),str(lenlc)])
        ymin,ymax = ax_lc.get_ylim()
        ax_lc.set_yticks(n.linspace(ymin,ymax,3).round(2))

        # image
        ax = fig.add_subplot(223)
        fov = n.degrees(1./d['uvres'])*60.
        ax.scatter(((xpix/2-srcra[0])-0.05*xpix)*fov/xpix, (ypix/2-srcdec[0])*fov/ypix, s=80, marker='<', facecolor='none')
        ax.scatter(((xpix/2-srcra[0])+0.05*xpix)*fov/xpix, (ypix/2-srcdec[0])*fov/ypix, s=80, marker='>', facecolor='none')
        impl = ax.imshow(im.transpose(), aspect='equal', origin='upper', interpolation='nearest', extent=[fov/2, -fov/2, -fov/2, fov/2], cmap=plt.get_cmap('Greys'), vmin=0, vmax=0.5*im.max())
        ax.set_xlabel('RA Offset (arcmin)')
        ax.set_ylabel('Dec Offset (arcmin)')

        if not outname:
            outname = os.path.join(d['workdir'], 'cands_%s_sc%dseg%di%ddm%ddt%d.png' % (d['fileroot'], scan, segment, loc[candnum, intcol], dmind, dtind))
            canvas = FigureCanvasAgg(fig)
            canvas.print_figure(outname)

        return ([],[])

def inspect_cand(mergepkl, snrmin=None, candnum=-1, scan=0, **kwargs):
    """ Create detailed plot of a single candidate.
    Thresholds (as minimum), then provides list of candidates to select with candnum.
    scan can be used to define scan number with pre-merge pkl file.
    kwargs passed to rt.set_pipeline
    """

    d = pickle.load(open(mergepkl, 'r'))
    loc, prop = read_candidates(mergepkl)
    
    if not os.path.dirname(d['filename']):
        d['filename'] = os.path.join(d['workdir'], d['filename'])

    # feature columns
    if 'snr2' in d['features']:
        snrcol = d['features'].index('snr2')
    elif 'snr1' in d['features']:
        snrcol = d['features'].index('snr1')
    if 'l2' in d['features']:
        lcol = d['features'].index('l2')
    elif 'l1' in d['features']:
        lcol = d['features'].index('l1')
    if 'm2' in d['features']:
        mcol = d['features'].index('m2')
    elif 'm1' in d['features']:
        mcol = d['features'].index('m1')
        
    if not scan:
        scancol = d['featureind'].index('scan')
    segmentcol = d['featureind'].index('segment')
    intcol = d['featureind'].index('int')
    dtindcol = d['featureind'].index('dtind')
    dmindcol = d['featureind'].index('dmind')

    # sort and prep candidate list
    snrs = prop[:,snrcol]
    if isinstance(snrmin, type(None)):
        snrmin = min(snrs)
    sortord = snrs.argsort()
    snrinds = n.where(snrs[sortord] > snrmin)[0]
    loc = loc[sortord][snrinds]
    prop = prop[sortord][snrinds]

    if candnum < 0:
        for i in range(len(loc)):
            print i, loc[i], prop[i, snrcol]
        print 'Returning candidate (loc, snr) ...'
        return (loc, prop[:,snrcol])
    else:
        print 'Reproducing and visualizing candidate %d at %s with properties %s.' % (candnum, loc[candnum], prop[candnum])
        if not scan:
            scan = loc[candnum, scancol]
            nsegments = len(d['segmenttimesdict'][scan])
        else:
            nsegments = len(d['segmenttimes'])
        segment = loc[candnum, segmentcol]
        dmind = loc[candnum, dmindcol]
        dtind = loc[candnum, dtindcol]
        candint = loc[candnum, intcol]
        dmarrorig = d['dmarr']
        dtarrorig = d['dtarr']

        d2 = rt.set_pipeline(d['filename'], scan, fileroot=d['fileroot'], paramfile='rtparams.py', savecands=False, savenoise=False, nsegments=nsegments, **kwargs)
        im, data = rt.pipeline_reproduce(d2, segment, (candint, dmind, dtind))  # with candnum, pipeline will return cand image and data

        # calc source location
        peakl, peakm = n.where(im == im.max())
        xpix,ypix = im.shape
        l1 = (xpix/2. - peakl[0])/(xpix*d['uvres'])
        m1 = (ypix/2. - peakm[0])/(ypix*d['uvres'])
        pt_ra, pt_dec = d['radec']
        src_ra, src_dec = source_location(pt_ra, pt_dec, l1, m1)
        print 'Peak (RA, Dec):', src_ra, src_dec

        print 'Returning candidate d, im, data'
        return d2, im, data

def source_location(pt_ra, pt_dec, l1, m1):
    """ Takes phase center and src l,m in radians to get ra,dec of source.
    """
    import math

    srcra = n.degrees(pt_ra + l1/math.cos(pt_dec)); srcdec = n.degrees(pt_dec + m1)
    return deg2HMS(srcra, srcdec)

def deg2HMS(ra='', dec='', round=False):
    """ quick and dirty coord conversion. googled to find bdnyc.org.
    """
    RA, DEC, rs, ds = '', '', '', ''
    if dec:
        if str(dec)[0] == '-':
            ds, dec = '-', abs(dec)
        deg = int(dec)
        decM = abs(int((dec-deg)*60))
        if round:
            decS = int((abs((dec-deg)*60)-decM)*60)
        else:
            decS = (abs((dec-deg)*60)-decM)*60
        DEC = '{0}{1} {2} {3}'.format(ds, deg, decM, decS)
  
    if ra:
        if str(ra)[0] == '-':
            rs, ra = '-', abs(ra)
        raH = int(ra/15)
        raM = int(((ra/15)-raH)*60)
        if round:
            raS = int(((((ra/15)-raH)*60)-raM)*60)
        else:
            raS = ((((ra/15)-raH)*60)-raM)*60
        RA = '{0}{1} {2} {3}'.format(rs, raH, raM, raS)
  
    if ra and dec:
        return (RA, DEC)
    else:
        return RA or DEC

