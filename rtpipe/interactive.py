from scipy.special import erfinv
import numpy as n
import logging, pickle, os
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from bokeh.plotting import ColumnDataSource, figure, save, output_file, vplot, hplot
from bokeh.models import HoverTool, TapTool, OpenURL
from collections import OrderedDict 
from rtpipe.parsecands import read_noise, read_candidates


def plot_interactive(mergepkl, noisepkl=None, thresh=6.0, thresh_link=7.0, ignoret=None, savehtml=True, url_path='plots'):
    """ Backwards compatible function for making interactive candidate summary plot """

    data = readdata(mergepkl)
    circleinds = calcinds(data, thresh, ignoret)
    crossinds = calcinds(data, -1*thresh, ignoret)
    edgeinds = calcinds(data, thresh_link, ignoret)

    fileroot = mergepkl.rstrip('_merge.pkl').lstrip('cands_')

    logger.info('Total on target time: {} s'.format(calcontime(data, inds=circleinds+crossinds+edgeinds)))

    if noisepkl:
        noiseplot = plotnoise(noisepkl)
    else:
        noiseplot = None

    combined = plotall(data, circleinds=circleinds, crossinds=crossinds, edgeinds=edgeinds, htmlname=None, noiseplot=noiseplot, url_path=url_path, fileroot=fileroot)

    if savehtml:
        output_file(mergepkl.rstrip('.pkl') + '.html')
        save(combined)
    else:
        return combined


def plotall(data, circleinds=None, crossinds=None, edgeinds=None, htmlname=None, noiseplot=None, url_path='../plots', fileroot=None):
    """ Create interactive plot from data dictionary

    data has keys of snr, time, dm, sizes, key and more.
    Optional index arguments are used to filter full data set.
    This can be used to remove bad segments or apply different symbols to subsets.
    url_path is path difference to png files for taptool. ('../plots' for jupyter notebook, 'plots' for public page)
    fileroot is the sdm file name used as root for all png files.
    """

    # set up data dictionary
    if not circleinds: circleinds = range(len(data['snrs']))
    if edgeinds:  # to remove double refs
        logger.info('{} circles (positive, not linked) and {} edges (positive, linked)'.format(len(circleinds), len(edgeinds)))
        circleinds = list(set(circleinds) - set(edgeinds))

    source = ColumnDataSource(data = dict({(key, tuple([value[i] for i in circleinds])) 
                                           for (key, value) in data.iteritems()}))

    # set ranges
    datalen = len(data['dm'])
    inds = circleinds
    if crossinds: inds += crossinds
    if edgeinds: inds += edgeinds
    dm = [data['dm'][i] for i in inds]
    dm_min = min(min(dm), max(dm)/1.2)
    dm_max = max(max(dm), min(dm)*1.2)
    time = [data['time'][i] for i in inds]
    time_min = min(time)
    time_max = max(time)
    specstd = [data['specstd'][i] for i in inds]
    specstd_min = min(specstd)
    specstd_max = max(specstd)
    imkur = [data['imkur'][i] for i in inds]
    imkur_min = min(imkur)
    imkur_max = max(imkur)
    l1 = [data['l1'][i] for i in inds]
    l1_min = min(l1)
    l1_max = max(l1)
    m1 = [data['m1'][i] for i in inds]
    m1_min = min(m1)
    m1_max = max(m1)

    TOOLS = "hover,tap,pan,box_select,wheel_zoom,reset"

    # reset-time plot
    dmt = figure(plot_width=950, plot_height=500, toolbar_location="left", x_axis_label='Time (s; relative)',
                 y_axis_label='DM (pc/cm3)', x_range=(time_min, time_max), y_range=(dm_min, dm_max), 
                 webgl=True, tools=TOOLS)
    dmt.circle(source.data['time'], source.data['dm'], size=source.data['sizes'], line_color=None, fill_color=source.data['colors'], 
               fill_alpha=0.2)

    # image location plot
    loc = figure(plot_width=450, plot_height=400, toolbar_location="left", x_axis_label='l1 (rad)', y_axis_label='m1 (rad)',
                 x_range=(l1_min, l1_max), y_range=(m1_min,m1_max), tools=TOOLS, webgl=True)
    loc.circle(source.data['l1'], source.data['m1'], size=source.data['sizes'], line_color=None, fill_color=source.data['colors'],
               fill_alpha=0.2)

    # cand spectrum/image statistics plot
    stat = figure(plot_width=450, plot_height=400, toolbar_location="left", x_axis_label='Spectral std',
                  y_axis_label='Image kurtosis', x_range=(specstd_min, specstd_max), 
                  y_range=(imkur_min, imkur_max), tools=TOOLS, webgl=True)
    stat.circle(source.data['specstd'], source.data['imkur'], size=source.data['sizes'], line_color=None, fill_color=source.data['colors'],
                fill_alpha=0.2)

    # norm prob plot
    norm = figure(plot_width=450, plot_height=400, toolbar_location="left", x_axis_label='SNR observed',
                  y_axis_label='SNR expected', tools=TOOLS, webgl=True)
    norm.circle(source.data['snrs'], source.data['zs'], size=source.data['sizes'], line_color=None, fill_color=source.data['colors'], fill_alpha=0.2)

    # set up negative symbols, if indexes in place
    if crossinds:
        sourceneg = ColumnDataSource(data = dict({(key, tuple([value[i] for i in crossinds]))
                                                  for (key, value) in data.iteritems()}))
        dmt.cross(sourceneg.data['time'], sourceneg.data['dm'], size=sourceneg.data['sizes'], line_color=sourceneg.data['colors'], line_alpha=0.2)
        loc.cross(sourceneg.data['l1'], sourceneg.data['m1'], size=sourceneg.data['sizes'], line_color=sourceneg.data['colors'], line_alpha=0.2)
        stat.cross(sourceneg.data['specstd'], sourceneg.data['imkur'], size=sourceneg.data['sizes'], line_color=sourceneg.data['colors'], line_alpha=0.2)
        norm.cross(sourceneg.data['abssnr'], sourceneg.data['zs'], size=sourceneg.data['sizes'], line_color=sourceneg.data['colors'], line_alpha=0.2)

    if edgeinds:
        sourceedge = ColumnDataSource(data = dict({(key, tuple([value[i] for i in edgeinds]))
                                                   for (key, value) in data.iteritems()}))
        dmt.circle('time', 'dm', size='sizes', line_color='colors', fill_color='colors', source=sourceedge, line_alpha=0.5, fill_alpha=0.2)
        loc.circle('l1', 'm1', size='sizes', line_color='colors', fill_color='colors', source=sourceedge, line_alpha=0.5, fill_alpha=0.2)
        stat.circle('specstd', 'imkur', size='sizes', line_color='colors', fill_color='colors', source=sourceedge, line_alpha=0.5, fill_alpha=0.2)
        norm.circle('snrs', 'zs', size='sizes', line_color='colors', fill_color='colors', source=sourceedge, line_alpha=0.5, fill_alpha=0.2)

    # define hover and url behavior
    hover = dmt.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('SNR', '@snrs'), ('time', '@time'), ('key', '@key')])
    hover = loc.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('SNR', '@snrs'), ('time', '@time'), ('key', '@key')])
    hover = stat.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('SNR', '@snrs'), ('time', '@time'), ('key', '@key')])
    hover = norm.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('SNR', '@snrs'), ('time', '@time'), ('key', '@key')])
    if url_path and fileroot:
        url = '{}/cands_{}_sc@scan-seg@seg-i@candint-dm@dmind-dt@dtind.png'.format(url_path, fileroot)
        taptool = dmt.select(type=TapTool)
        taptool.callback = OpenURL(url=url)
        taptool = loc.select(type=TapTool)
        taptool.callback = OpenURL(url=url)    
        taptool = stat.select(type=TapTool)
        taptool.callback = OpenURL(url=url)    
        taptool = norm.select(type=TapTool)
        taptool.callback = OpenURL(url=url)

    # arrange plots
    top = hplot(vplot(dmt), width=950)
    middle = hplot(vplot(loc), vplot(stat), width=950)
    if noiseplot:
        bottom = hplot(vplot(norm), vplot(noiseplot), width=950)
    else:
        bottom = hplot(vplot(norm), width=950)
    combined = vplot(top, middle, bottom, width=950)

    if htmlname:
        output_file(htmlname)
        save(combined)
    else:
        return combined


def readdata(mergepkl, sizerange=(2,70)):
    """ Converts candidate data from merged pkl file to dictionary for bokeh """

    # get cands from pkl
    with open(mergepkl,'r') as pkl:
        d = pickle.load(pkl)
        cands = pickle.load(pkl)

    # define columns to extract
    if 'snr2' in d['features']:
        snrcol = d['features'].index('snr2')
    elif 'snr1' in d['features']:
        snrcol = d['features'].index('snr1')
    l1col = d['features'].index('l1')
    m1col = d['features'].index('m1')
    specstdcol = d['features'].index('specstd')
    imkurcol = d['features'].index('imkurtosis')
    dtindcol = d['featureind'].index('dtind')
    dmindcol = d['featureind'].index('dmind')
    intcol = d['featureind'].index('int')
    segmentcol = d['featureind'].index('segment')
    scancol = d['featureind'].index('scan')

    # define data to plot
    snrs = []
    abssnr = []
    dm = []
    l1 = []
    m1 = []
    time = []
    specstd = []
    imkur = []
    key = []
    for k in cands.iterkeys():
        snrs.append(cands[k][snrcol])
        abssnr.append(abs(cands[k][snrcol]))
        dm.append(d['dmarr'][k[dmindcol]])
        l1.append(cands[k][l1col])
        m1.append(cands[k][m1col])
        time.append(24*3600*d['segmenttimesdict'][k[scancol]][k[segmentcol],0] + d['inttime']*k[intcol])
        specstd.append(cands[k][specstdcol])
        imkur.append(cands[k][imkurcol])
        key.append(k)
    time = time - min(time)
    scan, seg, candint, dmind, dtind, beamnum = zip(*key)
    zs = normprob(d, snrs)
    sizes = calcsize(snrs)
    colors = colorsat(l1, m1)

    data = dict(snrs=snrs, dm=dm, l1=l1, m1=m1, time=time, specstd=specstd,
                imkur=imkur, scan=scan, seg=seg, candint=candint, dmind=dmind,
                dtind=dtind, sizes=sizes, colors=colors, key=key, zs=zs, abssnr=abssnr)
    return data


def findhight(data, ignoret=None, threshold=20):
    """ Find bad time ranges from distribution of candidates.

    ignoret is list of tuples [(t0, t1), (t2, t3)] defining ranges to ignore.
    threshold is made above std of candidate distribution in time.
    """

    time = n.sort(data['time'])

    ww = n.ones(len(time), dtype=bool)  # initialize pass filter
    if ignoret:
        for (t0, t1) in ignoret:
            ww = ww & n.where( (time < t0) | (time > t1), True, False )

    bins = n.round(time[ww]).astype('int')
    counts = n.bincount(bins)
    high = n.where(counts > n.median(counts) + threshold*counts.std())[0]

    if len(high):
        logger.info('High times above {} sigma:'.format(threshold))
        logger.info('Bin (sec) \t Count (per sec)')
        for hh in high:
            logger.info('{}    \t {}'.format(hh, counts[hh]))
    else:
        logger.info('No time bins with {} sigma excess'.format(threshold))

    return ignoret   # convenience


def calcinds(data, threshold, ignoret=None):
    """ Find indexes for data above (or below) given threshold. """

    # select by time, too

    inds = []
    for i in range(len(data['time'])):
        snr = data['snrs'][i]
        time = data['time'][i]
        if (threshold >= 0 and snr > threshold) or (threshold < 0 and snr < threshold):
            if ignoret:
                incl = [t0 for (t0, t1) in ignoret if n.round(time).astype(int) in range(t0,t1)]
                logger.debug('{} {} {} {}'.format(n.round(time).astype(int), t0, t1, incl))
                if not incl:
                    inds.append(i)
            else:
                inds.append(i)

    return inds


def calcontime(data, inds=None):
    """ Given indices of good times, calculate total time per scan with indices. """

    if not inds:
        inds = range(len(data['time']))
        logger.info('No indices provided. Assuming all are valid.')

    scans = set([data['scan'][i] for i in inds])
    total = 0.
    for scan in scans:
        time = [data['time'][i] for i in inds if data['scan'][i] == scan]
        total += max(time) - min(time)

    return total


def plotnoise(noisepkl):
    """ Merged noise pkl converted to interactive cumulative histogram """

    # noise histogram
    noises = read_noise(noisepkl)
    imnoise = n.sort(noises[4])
    frac = [float(count)/len(imnoise) for count in reversed(range(1, len(imnoise)+1))]
    noiseplot = figure(plot_width=450, plot_height=400, toolbar_location="left", x_axis_label='Noise image std',
                       y_axis_label='Cumulative fraction', tools='pan, wheel_zoom, reset')
    noiseplot.line(imnoise, frac)

    return noiseplot


def normprob(d, snrs, inds=None):
    """ Uses observed SNR distribution to calculate normal probability SNR

    Uses state dict to calculate number of trials.
    snrs is list of all snrs in distribution.
    Returns list of expected snr given each input value's frequency of occurrence via the normal probability assumption
    """

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
    logger.info('Calculating normal probability distribution for npix*nints*ndms*dtfactor = %d' % (ntrials))

    # calc normal quantile
    if not inds: inds = range(len(snrs))
    snrsortpos = []
    snrsortneg = []
    for i in inds:
        if snrs[i] > 0:
            snrsortpos.append(snrs[i])
        elif snrs[i] < 0:
            snrsortneg.append(abs(snrs[i]))

    snrsortpos = sorted(snrsortpos, reverse=True)
    snrsortneg = sorted(snrsortneg, reverse=True)

    zval = []
    for (i, snr) in enumerate(snrs):
        if snr >= 0 and i in inds:
            zval.append(Z(quan(ntrials, snrsortpos.index(snr)+1)))
        elif snr < 0 and i in inds:
            zval.append(Z(quan(ntrials, snrsortneg.index(abs(snr))+1)))
        else:
            zval.append(0)

    return zval


def calcsize(snrs, sizerange=(2,70), inds=None):
    """ Uses SNR to calculate symbol size.

    snrs is a list of floats for candidate significance.
    inds is an optional list of indexes to use to calculate symbol size.
    Scaling of symbol size min max set by sizerange tuple (min, max).
    """

    if inds:
        smax = max([abs(snrs[i]) for i in inds])
        smin = min([abs(snrs[i]) for i in inds])
    else:
        smax = max([abs(snr) for snr in snrs])
        smin = min([abs(snr) for snr in snrs])
    return [sizerange[0] + sizerange[1] * ((abs(snr) - smin)/(smax - smin))**3 for snr in snrs]

    
def colorsat(l,m):
    """ Returns color for given l,m
    Designed to look like a color wheel that is more saturated in middle.
    """

    lm = n.zeros(len(l), dtype='complex')
    lm.real = l
    lm.imag = m
    red = 0.5*(1+n.cos(n.angle(lm)))
    green = 0.5*(1+n.cos(n.angle(lm) + 2*3.14/3))
    blue = 0.5*(1+n.cos(n.angle(lm) - 2*3.14/3))
    amp = 256*n.abs(lm)/n.abs(lm).max()
    return ["#%02x%02x%02x" % (n.floor(amp[i]*red[i]), n.floor(amp[i]*green[i]), n.floor(amp[i]*blue[i])) for i in range(len(l))]
