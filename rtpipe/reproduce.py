import numpy as np
import rtpipe.RT as rt
import rtpipe.parseparams as pp
import rtpipe.parsecands as pc
import pickle, logging, os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def plot_cand(candsfile, candloc=[], candnum=-1, threshold=0, savefile=True, returndata=False, outname='', **kwargs):
    """ Reproduce detection of a single candidate for plotting or inspection.

    candsfile can be merge or single-scan cands pkl file. Difference defined by presence of scan in d['featureind'].
    candloc reproduces candidate at given location (scan, segment, integration, dmind, dtind, beamnum).
    candnum selects one to reproduce from ordered list
    threshold is min of sbs(SNR) used to filter candidates to select with candnum.
    savefile/outname define if/how to save png of candidate
    if returndata, (im, data) returned.
    kwargs passed to rt.set_pipeline
    """

    # get candidate info
    loc, prop = pc.read_candidates(candsfile)

    # define state dict and overload with user prefs
    d0 = pickle.load(open(candsfile, 'r'))
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

        # clean up d0 of superfluous keys
        params = pp.Params()  # will be used as input to rt.set_pipeline
        for key in d0.keys():
            if not hasattr(params, key) and 'memory_limit' not in key:
                _ = d0.pop(key)

        d0['npix'] = 0
        d0['uvres'] = 0
        d0['nsegments'] = 0
        d0['logfile'] = False
        # get cand data
        d = rt.set_pipeline(filename, scan, **d0)
        im, data = rt.pipeline_reproduce(d, candloc, product='imdata') # removed loc[candnum]

        # optionally plot
        if savefile:
            loclabel = scan, segment, candint, dmind, dtind, beamnum
            make_cand_plot(d, im, data, loclabel, outname=outname)

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
       

def make_cand_plot(d, im, data, loclabel, outname=''):
    """ Builds candidate plot.
    Expects phased, dedispersed data (cut out in time, dual-pol), image, and metadata
    loclabel is used to label the plot with (scan, segment, candint, dmind, dtind, beamnum).
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
