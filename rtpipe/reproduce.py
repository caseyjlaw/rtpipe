import numpy as np
import rtpipe.RT as rt
import rtpipe.parseparams as pp
import rtpipe.parsecands as pc
import pickle, logging, os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy.ma as ma # mask for plotting
import matplotlib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def plot_cand(candsfile, candloc=[], candnum=-1, threshold=0, savefile=True, returndata=False, outname='', newplot=True, **kwargs):
    """ Reproduce detection of a single candidate for plotting or inspection.

    candsfile can be merge or single-scan cands pkl file. Difference defined by presence of scan in d['featureind'].
    candloc reproduces candidate at given location (scan, segment, integration, dmind, dtind, beamnum).
    candnum selects one to reproduce from ordered list
    threshold is min of sbs(SNR) used to filter candidates to select with candnum.
    savefile/outname define if/how to save png of candidate
    if returndata, (im, data) returned.
    kwargs passed to rt.set_pipeline
    if newplot, then plot with the new candidate plot using bridget's version
    """

    # get candidate info
    loc, prop, d0 = pc.read_candidates(candsfile, returnstate=True)

    # define state dict and overload with user prefs
    for key in kwargs:
        logger.info('Setting %s to %s' % (key, kwargs[key]))
        d0[key] = kwargs[key]
    d0['logfile'] = False  # no need to save log

    # feature columns
    if 'snr2' in d0['features']:
        snrcol = d0['features'].index('snr2')
    elif 'snr1' in d0['features']:
        snrcol = d0['features'].index('snr1')
    if 'l2' in d0['features']:
        lcol = d0['features'].index('l2')
    elif 'l1' in d0['features']:
        lcol = d0['features'].index('l1')
    if 'm2' in d0['features']:
        mcol = d0['features'].index('m2')
    elif 'm1' in d0['features']:
        mcol = d0['features'].index('m1')

    try:
        scancol = d0['featureind'].index('scan')  # if merged pkl
    except ValueError:
        scancol = -1   # if single-scan pkl
    segmentcol = d0['featureind'].index('segment')
    intcol = d0['featureind'].index('int')
    dtindcol = d0['featureind'].index('dtind')
    dmindcol = d0['featureind'].index('dmind')

    # sort and prep candidate list
    snrs = prop[:, snrcol]
    select = np.where(np.abs(snrs) > threshold)[0]
    loc = loc[select]
    prop = prop[select]
    times = pc.int2mjd(d0, loc)
    times = times - times[0]

    # default case will print cand info
    if (candnum < 0) and (not len(candloc)):
        logger.info('Getting candidates...')
        logger.info('candnum: loc, SNR, DM (pc/cm3), time (s; rel)')
        for i in range(len(loc)):
            logger.info("%d: %s, %.1f, %.1f, %.1f" % (i, str(loc[i]), prop[i, snrcol], np.array(d0['dmarr'])[loc[i,dmindcol]], times[i]))
    else:  # if candnum or candloc provided, try to reproduce
        if (candnum >= 0) and not len(candloc):
            logger.info('Reproducing and visualizing candidate %d at %s with properties %s.' % (candnum, loc[candnum], prop[candnum]))
            dmarrorig = d0['dmarr']
            dtarrorig = d0['dtarr']
            if scancol >= 0:  # here we have a merge pkl
                scan = loc[candnum, scancol]
            else:   # a scan-based cands pkl
                scan = d0['scan']
            segment = loc[candnum, segmentcol]
            candint = loc[candnum, intcol]
            dmind = loc[candnum, dmindcol]
            dtind = loc[candnum, dtindcol]
            beamnum = 0
            candloc = (scan, segment, candint, dmind, dtind, beamnum)
        elif len(candloc) and (candnum < 0):
            assert len(candloc) == 6, 'candloc should be length 6 ( scan, segment, candint, dmind, dtind, beamnum ).'
            logger.info('Reproducing and visualizing candidate %d at %s' % (candnum, candloc))
            dmarrorig = d0['dmarr']
            dtarrorig = d0['dtarr']
            scan, segment, candint, dmind, dtind, beamnum = candloc
        else:
            raise Exception, 'Provide candnum or candloc, not both'

        # if working locally, set workdir appropriately. Can also be used in queue system with full path given.
        if not os.path.dirname(candsfile):
            d0['workdir'] = os.getcwd()
        else:
            d0['workdir'] = os.path.dirname(candsfile)
        filename = os.path.join(d0['workdir'], os.path.basename(d0['filename']))

        if d0.has_key('segmenttimesdict'):  # using merged pkl
            segmenttimes = d0['segmenttimesdict'][scan]
        else:
            segmenttimes = d0['segmenttimes']

        # clean up d0 of superfluous keys
        params = pp.Params()  # will be used as input to rt.set_pipeline
        for key in d0.keys():
            if not hasattr(params, key): # and 'memory_limit' not in key:
                _ = d0.pop(key)
        d0['npix'] = 0
        d0['uvres'] = 0
        d0['logfile'] = False

# this triggers redefinition of segment boundaries. memory optimization changed, so this is a problem.
#        d0['nsegments'] = 0
#        d0['scale_nsegments'] = 1.
        d0['segmenttimes'] = segmenttimes
        d0['nsegments'] = len(segmenttimes)

        # get cand data
        d = rt.set_pipeline(filename, scan, **d0)
        (vismem, immem) = rt.calc_memory_footprint(d)
        if 'memory_limit' in d:
            assert vismem+immem < d['memory_limit'], 'memory_limit defined, but nsegments must (for now) be set to initial values to properly reproduce candidate'

        im, data = rt.pipeline_reproduce(d, candloc, product='imdata') # removed loc[candnum]

        # optionally plot
        if savefile:
            loclabel = scan, segment, candint, dmind, dtind, beamnum
            if newplot:
                make_cand_plot(d, im, data, loclabel, version=2, snrs=snrs, outname=outname) 
            else:
                make_cand_plot(d, im, data, loclabel, version=1, outname=outname)

        # optionally return data
        if returndata:
            return (im, data)


def refine_cand(candsfile, candloc=[], threshold=0):
    """ Helper function to interact with merged cands file and refine analysis

    candsfile is merged pkl file
    candloc (scan, segment, candint, dmind, dtind, beamnum) is as above.
    if no candloc, then it prints out cands above threshold.
    """

    if not candloc:
        plot_cand(candsfile, candloc=[], candnum=-1, threshold=threshold,
                  savefile=False, returndata=False)
    else:
        d = pickle.load(open(candsfile, 'r'))
        cands = rt.pipeline_refine(d, candloc)

    return cands
       

def make_cand_plot(d, im, data, loclabel, version=2, snrs=[], outname=''):
    """ Builds a new candidate plot, distinct from the original plots produced by make_cand_plot.
    Expects phased, dedispersed data (cut out in time, dual-pol), image, and metadata

    version 2 is the new one (thanks to bridget andersen). version 1 is the initial one.
    loclabel is used to label the plot with (scan, segment, candint, dmind, dtind, beamnum).
    snrs is array for an (optional) SNR histogram plot.
    d are used to label the plots with useful information.
    """

    # given d, im, data, make plot
    logger.info('Plotting...')
    logger.debug('(image, data) shape: (%s, %s)' % (str(im.shape), str(data.shape)))

    assert len(loclabel) == 6, 'loclabel should have (scan, segment, candint, dmind, dtind, beamnum)'
    scan, segment, candint, dmind, dtind, beamnum = loclabel

    # calc source location
    snrmin = im.min()/im.std()
    snrmax = im.max()/im.std()
    if snrmax > -1*snrmin:
        l1, m1 = rt.calc_lm(d, im, minmax='max')
        snrobs = snrmax
    else:
        l1, m1 = rt.calc_lm(d, im, minmax='min')
        snrobs = snrmin
    pt_ra, pt_dec = d['radec']
    src_ra, src_dec = source_location(pt_ra, pt_dec, l1, m1)
    logger.info('Peak (RA, Dec): %s, %s' % (src_ra, src_dec))

    # convert l1 and m1 from radians to arcminutes
    l1arcm = l1*180.*60./np.pi
    m1arcm = m1*180.*60./np.pi

    if version == 1:
        # build plot
        fig = plt.Figure(figsize=(8.5,8))
        ax = fig.add_subplot(221, axisbg='white')

        # add annotating info
        ax.text(0.1, 0.9, d['fileroot'], fontname='sans-serif', transform = ax.transAxes)
        ax.text(0.1, 0.8, 'sc %d, seg %d, int %d, DM %.1f, dt %d' % (scan, segment, candint, d['dmarr'][dmind], d['dtarr'][dtind]), fontname='sans-serif', transform = ax.transAxes)
        ax.text(0.1, 0.7, 'Peak: (' + str(np.round(l1, 3)) + ' ,' + str(np.round(m1, 3)) + '), SNR: ' + str(np.round(snrobs, 1)), fontname='sans-serif', transform = ax.transAxes)

        # plot dynamic spectra
        left, width = 0.6, 0.2
        bottom, height = 0.2, 0.7
        rect_dynsp = [left, bottom, width, height]
        rect_lc = [left, bottom-0.1, width, 0.1]    
        rect_sp = [left+width, bottom, 0.1, height]
        ax_dynsp = fig.add_axes(rect_dynsp)
        ax_lc = fig.add_axes(rect_lc)    
        ax_sp = fig.add_axes(rect_sp)
        spectra = np.swapaxes(data.real,0,1)      # seems that latest pickle actually contains complex values in spectra...
        dd = np.concatenate( (spectra[...,0], np.zeros_like(spectra[...,0]), spectra[...,1]), axis=1)    # make array for display with white space between two pols
        impl = ax_dynsp.imshow(dd, origin='lower', interpolation='nearest', aspect='auto', cmap=plt.get_cmap('Greys'))
        ax_dynsp.text(0.5, 0.95, 'RR LL', horizontalalignment='center', verticalalignment='center', fontsize=16, color='w', transform = ax_dynsp.transAxes)
        ax_dynsp.set_yticks(range(0,len(d['freq']),30))
        ax_dynsp.set_yticklabels(d['freq'][::30])
        ax_dynsp.set_ylabel('Freq (GHz)')
        ax_dynsp.set_xlabel('Integration (rel)')
        spectrum = spectra[:,len(spectra[0])/2].mean(axis=1)      # assume pulse in middle bin. get stokes I spectrum. **this is wrong in a minority of cases.**
        ax_sp.plot(spectrum, range(len(spectrum)), 'k.')
        ax_sp.plot(np.zeros(len(spectrum)), range(len(spectrum)), 'k:')
        ax_sp.set_ylim(0, len(spectrum))
        ax_sp.set_yticklabels([])
        xmin,xmax = ax_sp.get_xlim()
        ax_sp.set_xticks(np.linspace(xmin,xmax,3).round(2))
        ax_sp.set_xlabel('Flux (Jy)')
        lc = dd.mean(axis=0)
        lenlc = len(data)  # old (stupid) way: lenlc = np.where(lc == 0)[0][0]
        ax_lc.plot(range(0,lenlc)+range(2*lenlc,3*lenlc), list(lc)[:lenlc] + list(lc)[-lenlc:], 'k.')
        ax_lc.plot(range(0,lenlc)+range(2*lenlc,3*lenlc), list(np.zeros(lenlc)) + list(np.zeros(lenlc)), 'k:')
        ax_lc.set_xlabel('Integration')
        ax_lc.set_ylabel('Flux (Jy)')
        ax_lc.set_xticks([0,0.5*lenlc,lenlc,1.5*lenlc,2*lenlc,2.5*lenlc,3*lenlc])
        ax_lc.set_xticklabels(['0',str(lenlc/2),str(lenlc),'','0',str(lenlc/2),str(lenlc)])
        ymin,ymax = ax_lc.get_ylim()
        ax_lc.set_yticks(np.linspace(ymin,ymax,3).round(2))

        # image
        ax = fig.add_subplot(223)
        fov = np.degrees(1./d['uvres'])*60.
        impl = ax.imshow(im.transpose(), aspect='equal', origin='upper',
                         interpolation='nearest', extent=[fov/2, -fov/2, -fov/2, fov/2],
                         cmap=plt.get_cmap('Greys'), vmin=0, vmax=0.5*im.max())
        ax.set_xlabel('RA Offset (arcmin)')
        ax.set_ylabel('Dec Offset (arcmin)')

    elif version == 2:
        # build overall plot
        fig = plt.Figure(figsize=(12.75,8))
    
        # add metadata in subfigure
        ax = fig.add_subplot(2,3,1, axisbg='white')   
    
        # calculate the overall dispersion delay: dd
        f1 = d['freq_orig'][0]
        f2 = d['freq_orig'][len(d['freq_orig'])-1]
        dd = 4.15*d['dmarr'][dmind]*(f1**(-2)-f2**(-2))

        # add annotating info
        start = 1.1 # these values determine the spacing and location of the annotating information
        space = 0.07
        left = 0.0
        ax.text(left, start, d['fileroot'], fontname='sans-serif', transform = ax.transAxes, fontsize='small')
        ax.text(left, start-space, 'Peak (arcmin): (' + str(np.round(l1arcm, 3)) + ', ' + str(np.round(m1arcm, 3)) + ')', fontname='sans-serif', transform = ax.transAxes, fontsize='small')
        # split the RA and Dec and display in a nice format
        ra = src_ra.split()
        dec = src_dec.split()
        ax.text(left, start-2*space, 'Peak (RA, Dec): (' + ra[0] + ':' + ra[1] + ':' + ra[2][0:4] + ', ' + dec[0] + ':' + dec[1] + ':' + dec[2][0:4]  + ')', 
                fontname='sans-serif', transform = ax.transAxes, fontsize='small')
        ax.text(left, start-3*space, 'Source: ' + str(d['source']), fontname='sans-serif', transform = ax.transAxes, fontsize='small')
        ax.text(left, start-4*space, 'scan: ' + str(scan), fontname='sans-serif', transform = ax.transAxes, fontsize='small')
        ax.text(left, start-5*space, 'segment: ' + str(segment), fontname='sans-serif', transform = ax.transAxes, fontsize='small')
        ax.text(left, start-6*space, 'integration: ' + str(candint), fontname='sans-serif', transform = ax.transAxes, fontsize='small')
        ax.text(left, start-7*space, 'DM = ' + str(d['dmarr'][dmind]) + ' (index ' + str(dmind) + ')', fontname='sans-serif', transform = ax.transAxes, fontsize='small')
        ax.text(left, start-8*space, 'dt = ' + str(np.round(d['inttime']*d['dtarr'][dtind], 3)*1e3) + ' ms' + ' (index ' + str(dtind) + ')', fontname='sans-serif', transform = ax.transAxes, fontsize='small')
        ax.text(left, start-9*space, 'disp delay = ' + str(np.round(dd,1)) + ' ms', fontname='sans-serif', transform = ax.transAxes, fontsize='small')
        ax.text(left, start-10*space, 'SNR: ' + str(np.round(snrobs, 1)), fontname='sans-serif', transform = ax.transAxes, fontsize='small')
        # set the plot invisible so that it doesn't interfere with annotations
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white') 
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')    

        # plot full dynamic spectra
        left, width = 0.75, 0.2*2./3.
        bottom, height = 0.2, 0.7
        rect_dynsp1 = [left, bottom, width/3., height] # three rectangles for each panel of the spectrum (RR, RR+LL, LL)
        rect_dynsp2 = [left+width/3., bottom, width/3., height]
        rect_dynsp3 = [left+2.*width/3., bottom, width/3., height]
        rect_lc1 = [left, bottom-0.1, width/3., 0.1] 
        rect_lc2 = [left+width/3., bottom-0.1, width/3., 0.1]
        rect_lc3 = [left+2.*width/3., bottom-0.1, width/3., 0.1]
        rect_sp = [left+width, bottom, 0.1*2./3., height]
        ax_dynsp1 = fig.add_axes(rect_dynsp1)
        ax_dynsp2 = fig.add_axes(rect_dynsp2, sharey=ax_dynsp1) # sharey so that axes line up
        ax_dynsp3 = fig.add_axes(rect_dynsp3, sharey=ax_dynsp1)
        # make RR+LL and LL dynamic spectra y labels invisible so they don't interfere with the plots
        [label.set_visible(False) for label in ax_dynsp2.get_yticklabels()]
        [label.set_visible(False) for label in ax_dynsp3.get_yticklabels()]
        ax_sp = fig.add_axes(rect_sp, sharey=ax_dynsp3)
        [label.set_visible(False) for label in ax_sp.get_yticklabels()]
        ax_lc1 = fig.add_axes(rect_lc1)
        ax_lc2 = fig.add_axes(rect_lc2, sharey=ax_lc1)
        ax_lc3 = fig.add_axes(rect_lc3, sharey=ax_lc1)
        [label.set_visible(False) for label in ax_lc2.get_yticklabels()]
        [label.set_visible(False) for label in ax_lc3.get_yticklabels()]
    
        # now actually plot the data
        spectra = np.swapaxes(data.real,0,1)     
        dd1 = spectra[...,0]
        dd2 = spectra[...,0] + spectra[...,1]
        dd3 = spectra[...,1]
        colormap = 'viridis'
        impl1 = ax_dynsp1.imshow(dd1, origin='lower', interpolation='nearest', aspect='auto', cmap=plt.get_cmap(colormap))
        impl2 = ax_dynsp2.imshow(dd2, origin='lower', interpolation='nearest', aspect='auto', cmap=plt.get_cmap(colormap))
        impl3 = ax_dynsp3.imshow(dd3, origin='lower', interpolation='nearest', aspect='auto', cmap=plt.get_cmap(colormap))
        ax_dynsp1.set_yticks(range(0,len(d['freq']),30))
        ax_dynsp1.set_yticklabels(d['freq'][::30])
        ax_dynsp1.set_ylabel('Freq (GHz)')
        ax_dynsp1.set_xlabel('RR')
        ax_dynsp1.xaxis.set_label_position('top')
        ax_dynsp2.set_xlabel('RR+LL')
        ax_dynsp2.xaxis.set_label_position('top')
        ax_dynsp3.set_xlabel('LL')
        ax_dynsp3.xaxis.set_label_position('top')
        [label.set_visible(False) for label in ax_dynsp1.get_xticklabels()] # set xlabels invisible so that they don't interefere with lc plots
        ax_dynsp1.get_yticklabels()[0].set_visible(False) # This one y label was getting in the way
    
        # plot stokes I spectrum of the candidate pulse (assume middle bin)
        spectrum = spectra[:,len(spectra[0])/2].mean(axis=1) # select stokes I middle bin
        ax_sp.plot(spectrum, range(len(spectrum)), 'k.')
        ax_sp.plot(np.zeros(len(spectrum)), range(len(spectrum)), 'r:') # plot 0 Jy dotted line
        xmin,xmax = ax_sp.get_xlim()
        ax_sp.set_xticks(np.linspace(xmin,xmax,3).round(2))
        ax_sp.set_xlabel('Flux (Jy)')    

        # plot mean flux values for each time bin
        lc1 = dd1.mean(axis=0)
        lc2 = dd2.mean(axis=0)
        lc3 = dd3.mean(axis=0)
        lenlc = len(data)
        ax_lc1.plot(range(0,lenlc), list(lc1)[:lenlc], 'k.')
        ax_lc2.plot(range(0,lenlc), list(lc2)[:lenlc], 'k.')
        ax_lc3.plot(range(0,lenlc), list(lc3)[:lenlc], 'k.')
        ax_lc1.plot(range(0,lenlc), list(np.zeros(lenlc)), 'r:') # plot 0 Jy dotted line for each plot
        ax_lc2.plot(range(0,lenlc), list(np.zeros(lenlc)), 'r:')
        ax_lc3.plot(range(0,lenlc), list(np.zeros(lenlc)), 'r:')    
        ax_lc2.set_xlabel('Integration (rel)')
        ax_lc1.set_ylabel('Flux (Jy)')
        ax_lc1.set_xticks([0, 0.5*lenlc, lenlc])
        ax_lc1.set_xticklabels(['0', str(lenlc/2), str(lenlc)]) # note I chose to only show the '0' label for one of the plots to avoid messy overlap
        ax_lc2.set_xticks([0, 0.5*lenlc, lenlc])
        ax_lc2.set_xticklabels(['', str(lenlc/2), str(lenlc)])
        ax_lc3.set_xticks([0, 0.5*lenlc, lenlc])
        ax_lc3.set_xticklabels(['', str(lenlc/2), str(lenlc)])
        ymin,ymax = ax_lc1.get_ylim()
        ax_lc1.set_yticks(np.linspace(ymin,ymax,3).round(2))

        # readjust the x tick marks on the dynamic spectra so that they line up with the lc plots
        ax_dynsp1.set_xticks([0, 0.5*lenlc, lenlc])
        ax_dynsp2.set_xticks([0, 0.5*lenlc, lenlc])
        ax_dynsp3.set_xticks([0, 0.5*lenlc, lenlc])

        # plot second set of dynamic spectra (averaged across frequency bins to get SNR=2 for the detected candidate)
        left, width = 0.45, 0.1333
        bottom, height = 0.1, 0.4
        rect_dynsp1 = [left, bottom, width/3., height]
        rect_dynsp2 = [left+width/3., bottom, width/3., height]
        rect_dynsp3 = [left+2.*width/3., bottom, width/3., height]
        rect_sp = [left+width, bottom, 0.1*2./3., height]
        ax_dynsp1 = fig.add_axes(rect_dynsp1)
        ax_dynsp2 = fig.add_axes(rect_dynsp2, sharey=ax_dynsp1)
        ax_dynsp3 = fig.add_axes(rect_dynsp3, sharey=ax_dynsp1)
        # make RR+LL and LL dynamic spectra y labels invisible so they don't interfere with the plots
        [label.set_visible(False) for label in ax_dynsp2.get_yticklabels()]
        [label.set_visible(False) for label in ax_dynsp3.get_yticklabels()]
        ax_sp = fig.add_axes(rect_sp, sharey=ax_dynsp3)
        [label.set_visible(False) for label in ax_sp.get_yticklabels()]

        # calculate the number of frequency rows to average together (make the plot have an SNR of 2)
        n = int((2.*(len(spectra))**0.5/snrobs)**2)
        if n == 0: # if n==0 then don't average any (avoids errors for modding and dividing by 0)
            dd1avg = dd1
            dd3avg = dd3
        else:
            # otherwise, add zeros onto the data so that it's length is cleanly divisible by n (makes it easier to average over)
            dd1zerotemp = np.concatenate((np.zeros((n-len(spectra)%n, len(spectra[0])), dtype=dd1.dtype), dd1), axis=0)
            dd3zerotemp = np.concatenate((np.zeros((n-len(spectra)%n, len(spectra[0])), dtype=dd3.dtype), dd3), axis=0)
            # make them masked arrays so that the appended zeros do not affect average calculation
            zeros = np.zeros((len(dd1), len(dd1[0])))
            ones = np.ones((n-len(spectra)%n, len(dd1[0])))
            masktemp = np.concatenate((ones, zeros), axis=0)
            dd1zero = ma.masked_array(dd1zerotemp, mask=masktemp)
            dd3zero = ma.masked_array(dd3zerotemp, mask=masktemp)
            # average together the data
            dd1avg = np.array([], dtype=dd1.dtype)
            for i in range(len(spectra[0])):
                temp = dd1zero[:,i].reshape(-1,n)
                tempavg = np.reshape(np.mean(temp, axis=1),(len(temp),1))
                temprep = np.repeat(tempavg, n, axis=0) # repeats the mean values to create more pixels (easier to properly crop when it is finally displayed)
                if i == 0:
                    dd1avg = temprep
                else:
                    dd1avg = np.concatenate((dd1avg, temprep), axis=1)
            dd3avg = np.array([], dtype=dd3.dtype)
            for i in range(len(spectra[0])):
                temp = dd3zero[:,i].reshape(-1,n)
                tempavg = np.reshape(np.mean(temp, axis=1),(len(temp),1))
                temprep = np.repeat(tempavg, n, axis=0)
                if i == 0:
                    dd3avg = temprep
                else:
                    dd3avg = np.concatenate((dd3avg, temprep), axis=1)
        dd2avg = dd1avg + dd3avg # add together to get averaged RR+LL spectrum
        colormap = 'viridis'
        if n == 0: # again, if n==0 then don't crop the spectra because no zeroes were appended
            dd1avgcrop = dd1avg
            dd2avgcrop = dd2avg
            dd3avgcrop = dd3avg
        else: # otherwise, crop off the appended zeroes
            dd1avgcrop = dd1avg[len(ones):len(dd1avg),:]
            dd2avgcrop = dd2avg[len(ones):len(dd2avg),:]
            dd3avgcrop = dd3avg[len(ones):len(dd3avg),:]
        impl1 = ax_dynsp1.imshow(dd1avgcrop, origin='lower', interpolation='nearest', aspect='auto', cmap=plt.get_cmap(colormap))
        impl2 = ax_dynsp2.imshow(dd2avgcrop, origin='lower', interpolation='nearest', aspect='auto', cmap=plt.get_cmap(colormap))
        impl3 = ax_dynsp3.imshow(dd3avgcrop, origin='lower', interpolation='nearest', aspect='auto', cmap=plt.get_cmap(colormap))
        ax_dynsp1.set_yticks(range(0,len(d['freq']), 30))
        ax_dynsp1.set_yticklabels(d['freq'][::30])
        ax_dynsp1.set_ylabel('Freq (GHz)')
        ax_dynsp1.set_xlabel('RR')
        ax_dynsp1.xaxis.set_label_position('top')
        ax_dynsp2.set_xlabel('Integration (rel)')
        ax2 = ax_dynsp2.twiny()
        ax2.set_xlabel('RR+LL')
        [label.set_visible(False) for label in ax2.get_xticklabels()]
        ax_dynsp3.set_xlabel('LL')
        ax_dynsp3.xaxis.set_label_position('top')
    
        # plot stokes I spectrum of the candidate pulse in the averaged data (assume middle bin)
        ax_sp.plot(dd2avgcrop[:,len(dd2avgcrop[0])/2]/2., range(len(dd2avgcrop)), 'k.')
        ax_sp.plot(np.zeros(len(dd2avgcrop)), range(len(dd2avgcrop)), 'r:')
        xmin,xmax = ax_sp.get_xlim()
        ax_sp.set_xticks(np.linspace(xmin,xmax,3).round(2))
        ax_sp.get_xticklabels()[0].set_visible(False)
        ax_sp.set_xlabel('Flux (Jy)')    

        # readjust the x tick marks on the dynamic spectra
        ax_dynsp1.set_xticks([0, 0.5*lenlc, lenlc])
        ax_dynsp1.set_xticklabels(['0', str(lenlc/2), str(lenlc)])
        ax_dynsp2.set_xticks([0, 0.5*lenlc, lenlc])
        ax_dynsp2.set_xticklabels(['', str(lenlc/2), str(lenlc)])
        ax_dynsp3.set_xticks([0, 0.5*lenlc, lenlc])
        ax_dynsp3.set_xticklabels(['', str(lenlc/2), str(lenlc)])

        # plot the image and zoomed cutout
        ax = fig.add_subplot(2,3,4)
        fov = np.degrees(1./d['uvres'])*60.
        impl = ax.imshow(im.transpose(), aspect='equal', origin='upper',
                         interpolation='nearest', extent=[fov/2, -fov/2, -fov/2, fov/2],
                         cmap=plt.get_cmap('viridis'), vmin=0, vmax=0.5*im.max())
        ax.set_xlabel('RA Offset (arcmin)')
        ax.set_ylabel('Dec Offset (arcmin)')
        ax.autoscale(False) # to stop the plot from autoscaling when we plot the triangles that label the location
        # add markers on the axes to indicate the measured position of the candidate
        ax.scatter(x=[l1arcm], y=[-fov/2], c='#ffff00', s=60, marker='^', clip_on=False)
        ax.scatter(x=[fov/2], y=[m1arcm], c='#ffff00', s=60, marker='>', clip_on=False)
        ax.set_frame_on(False) # makes it so the axis does not intersect the location triangles (for cosmetic reasons)
    
        # add a zoomed cutout image of the candidate (set width at 5*synthesized beam)
        key = d['vrange'].keys()[0]
        umax = d['urange'][key]
        vmax = d['vrange'][key]
        uvdist = (umax**2+vmax**2)**0.5
        sbeam = np.degrees(d['uvoversample']/uvdist)*60. # calculate synthesized beam in arcminutes
        # figure out the location to center the zoomed image on
        xratio = len(im[0])/fov # pix/arcmin
        yratio = len(im)/fov # pix/arcmin
        mult = 5 # sets how many times the synthesized beam the zoomed FOV is
        xmin = int(len(im[0])/2-(m1arcm+sbeam*mult)*xratio)
        xmax = int(len(im[0])/2-(m1arcm-sbeam*mult)*xratio)
        ymin = int(len(im)/2-(l1arcm+sbeam*mult)*yratio)
        ymax = int(len(im)/2-(l1arcm-sbeam*mult)*yratio)
        left, width = 0.231, 0.15
        bottom, height = 0.465, 0.15
        rect_imcrop = [left, bottom, width, height]
        ax_imcrop = fig.add_axes(rect_imcrop)
        impl = ax_imcrop.imshow(im.transpose()[xmin:xmax,ymin:ymax], aspect=1, origin='upper',
                                interpolation='nearest', extent=[-1,1,-1,1],
                                cmap=plt.get_cmap('viridis'), vmin=0, vmax=0.5*im.max())
        # setup the axes
        ax_imcrop.set_ylabel('Dec (arcmin)')
        ax_imcrop.set_xlabel('RA (arcmin)')
        ax_imcrop.xaxis.set_label_position('top')
        ax_imcrop.xaxis.tick_top()
        xlabels = [str(np.round(l1arcm+sbeam*mult/2, 1)), '', str(np.round(l1arcm, 1)), '', str(np.round(l1arcm-sbeam*mult/2, 1))]
        ylabels = [str(np.round(m1arcm-sbeam*mult/2, 1)), '', str(np.round(m1arcm, 1)), '', str(np.round(m1arcm+sbeam*mult/2, 1))]
        ax_imcrop.set_xticklabels(xlabels)
        ax_imcrop.set_yticklabels(ylabels)
        # change axis label location of inset so it doesn't interfere with the full picture
        ax_imcrop.get_yticklabels()[0].set_verticalalignment('bottom')

        # create SNR versus N histogram for the whole observation (properties for each candidate in the observation given by prop)
        if len(snrs):
            left, width = 0.45, 0.2
            bottom, height = 0.6, 0.3
            rect_snr = [left, bottom, width, height]
            ax_snr = fig.add_axes(rect_snr)
            pos_snrs = snrs[snrs >= 0]
            neg_snrs = snrs[snrs < 0]
            if not len(neg_snrs):  # if working with subset and only positive snrs
                neg_snrs = pos_snrs
                nonegs = True
            else:
                nonegs = False
            minval = 5.5
            maxval = 8.0
            # determine the min and max values of the x axis
            if min(pos_snrs) < min(np.abs(neg_snrs)):
                minval = min(pos_snrs)
            else:
                minval = min(np.abs(neg_snrs))
            if max(pos_snrs) > max(np.abs(neg_snrs)):
                maxval = max(pos_snrs)
            else:
                maxval = max(np.abs(neg_snrs))

            # positive SNR bins are in blue
            # absolute values of negative SNR bins are taken and plotted as red x's on top of positive blue bins for compactness
            n, b, patches = ax_snr.hist(pos_snrs, 50, (minval,maxval), facecolor='blue', zorder=1)
            vals, bin_edges = np.histogram(np.abs(neg_snrs), 50, (minval,maxval))
            bins = np.array([(bin_edges[i]+bin_edges[i+1])/2. for i in range(len(vals))])
            vals = np.array(vals)
            if not nonegs:
                ax_snr.scatter(bins[vals > 0], vals[vals > 0], marker='x', c='orangered', alpha=1.0, zorder=2)
            ax_snr.set_xlabel('SNR')
            ax_snr.set_xlim(left=minval-0.2)
            ax_snr.set_xlim(right=maxval+0.2)
            ax_snr.set_ylabel('N')
            ax_snr.set_yscale('log')
            # draw vertical line where the candidate SNR is
            ax_snr.axvline(x=np.abs(snrobs), linewidth=1, color='y', alpha=0.7)

    else:
        logger.warn('make_cand_plot version not recognized.')

    if not outname:
        outname = os.path.join(d['workdir'],
                               'cands_{}_sc{}-seg{}-i{}-dm{}-dt{}.png'.format(d['fileroot'], scan,
                                                                              segment, candint, dmind, dtind))

    try:
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(outname)
    except ValueError:
        logger.warn('Could not write figure to %s' % outname)


def convertloc(candsfile, candloc, memory_limit):
    """ For given state and location that are too bulky, calculate new location given memory_limit. """

    scan, segment, candint, dmind, dtind, beamnum = candloc

    # set up state and find absolute integration of candidate
    d0 = pickle.load(open(candsfile, 'r'))
    filename = os.path.basename(d0['filename'])
    readints0 = d0['readints']
    nskip0 = (24*3600*(d0['segmenttimes'][segment, 0]
                      - d0['starttime_mjd'])
             / d0['inttime']).astype(int)
    candint_abs = nskip0 + candint

    logger.debug('readints0 {} nskip0 {}, candint_abs {}'.format(readints0, nskip0, candint_abs))

    # clean up d0 and resubmit to set_pipeline
    params = pp.Params()
    for key in d0.keys():
        if not hasattr(params, key):
            _ = d0.pop(key)
    d0['logfile'] = False
    d0['npix'] = 0
    d0['uvres'] = 0
    d0['nsegments'] = 0
    d0['memory_limit'] = memory_limit
    d = rt.set_pipeline(os.path.basename(filename), scan, **d0)

    # find best segment for new state
    readints = d['readints']
    nskips = [(24*3600*(d['segmenttimes'][segment, 0]
                        - d['starttime_mjd']) / d['inttime']).astype(int)
              for segment in range(d['nsegments'])]

    posind = [i for i in range(len(nskips)) if candint_abs - nskips[i] > 0]
    segment_new = [seg for seg in posind if candint_abs - nskips[seg] == min([candint_abs - nskips[i] for i in posind])][0]
    candint_new = candint_abs - nskips[segment_new]

    logger.debug('nskips {}, segment_new {}'.format(nskips, segment_new))

    return [scan, segment_new, candint_new, dmind, dtind, beamnum]


def source_location(pt_ra, pt_dec, l1, m1):
    """ Takes phase center and src l,m in radians to get ra,dec of source.
    Returns string ('hh mm ss', 'dd mm ss')
    """
    import math

    srcra = np.degrees(pt_ra + l1/math.cos(pt_dec))
    srcdec = np.degrees(pt_dec + m1)

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
