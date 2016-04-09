from scipy.special import erfinv
import numpy as np
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


def plotall(data, circleinds=[], crossinds=[], edgeinds=[], htmlname=None, noiseplot=None, url_path='plots', fileroot=None):
    """ Create interactive plot from data dictionary

    data has keys of snr, time, dm, sizes, key and more.
    Optional index arguments are used to filter full data set.
    This can be used to remove bad segments or apply different symbols to subsets.
    url_path is path difference to png files for taptool. ('../plots' for jupyter notebook, 'plots' for public page)
    fileroot is the sdm file name used as root for all png files.
    """

    # set up data dictionary
    if not circleinds: circleinds = range(len(data['snrs']))

    # set ranges
    datalen = len(data['dm'])
    inds = circleinds + crossinds + edgeinds
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

    # dm-time plot
    dmt = plotdmt(data, circleinds=circleinds, crossinds=crossinds, edgeinds=edgeinds, url_path=url_path, fileroot=fileroot, tools=TOOLS) # maybe add size?

    # image location plot
    loc = plotloc(data, circleinds=circleinds, crossinds=crossinds, edgeinds=edgeinds, url_path=url_path, fileroot=fileroot, tools=TOOLS)

    # cand spectrum/image statistics plot
    stat = plotstat(data, circleinds=circleinds, crossinds=crossinds, edgeinds=edgeinds, url_path=url_path, fileroot=fileroot, tools=TOOLS)

    # norm prob plot
    norm = plotnorm(data, circleinds=circleinds, crossinds=crossinds, edgeinds=edgeinds, url_path=url_path, fileroot=fileroot, tools=TOOLS)

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


def plotdmt(data, circleinds=[], crossinds=[], edgeinds=[], url_path=None, fileroot=None, tools="hover,tap,pan,box_select,wheel_zoom,reset"):
    """ Make a dm-time figure """

    if not circleinds: circleinds = range(len(data['snrs']))

    # set ranges
    datalen = len(data['dm'])
    inds = circleinds + crossinds + edgeinds
    dm = [data['dm'][i] for i in inds]
    dm_min = min(min(dm), max(dm)/1.2)
    dm_max = max(max(dm), min(dm)*1.2)
    time = [data['time'][i] for i in inds]
    time_min = min(time)
    time_max = max(time)

    source = ColumnDataSource(data = dict({(key, tuple([value[i] for i in circleinds if i not in edgeinds])) 
                                           for (key, value) in data.iteritems()}))
    dmt = figure(plot_width=950, plot_height=500, toolbar_location="left", x_axis_label='Time (s; relative)',
                 y_axis_label='DM (pc/cm3)', x_range=(time_min, time_max), y_range=(dm_min, dm_max), 
                 webgl=True, tools=tools)
    dmt.circle('time', 'dm', size='sizes', fill_color='colors', line_color=None, fill_alpha=0.2, source=source)
#    dmt.circle(source.data['time'], source.data['dm'], size=source.data['sizes'], fill_color=source.data['colors'], line_color=None, fill_alpha=0.2)

    if crossinds:
        sourceneg = ColumnDataSource(data = dict({(key, tuple([value[i] for i in crossinds]))
                                                  for (key, value) in data.iteritems()}))
        dmt.cross('time', 'dm', size='sizes', fill_color='colors', line_alpha=0.2, source=sourceneg)
#        dmt.cross(sourceneg.data['time'], sourceneg.data['dm'], size=sourceneg.data['sizes'], fill_color=sourceneg.data['colors'], line_alpha=0.2)

    if edgeinds:
        sourceedge = ColumnDataSource(data = dict({(key, tuple([value[i] for i in edgeinds]))
                                                   for (key, value) in data.iteritems()}))
        dmt.circle('time', 'dm', size='sizes', line_color='colors', fill_color='colors', line_alpha=0.7, fill_alpha=0.2, source=sourceedge)
#        dmt.circle(sourceedge.data['time'], sourceedge.data['dm'], size=sourceedge.data['sizes'], line_color=sourceedge.data['colors'], fill_color=sourceedge.data['colors'], line_alpha=0.7, fill_alpha=0.2)
    hover = dmt.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('SNR', '@snrs'), ('time', '@time'), ('key', '@key')])

    if url_path and fileroot:
        url = '{}/cands_{}_sc@scan-seg@seg-i@candint-dm@dmind-dt@dtind.png'.format(url_path, fileroot)
        taptool = dmt.select(type=TapTool)
        taptool.callback = OpenURL(url=url)

    return dmt


def plotloc(data, circleinds=[], crossinds=[], edgeinds=[], url_path=None, fileroot=None, tools="hover,tap,pan,box_select,wheel_zoom,reset"):
    """ Make a loc figure """

    if not circleinds: circleinds = range(len(data['snrs']))

    # set ranges
    datalen = len(data['dm'])
    inds = circleinds + crossinds + edgeinds
    l1 = [data['l1'][i] for i in inds]
    l1_min = min(l1)
    l1_max = max(l1)
    m1 = [data['m1'][i] for i in inds]
    m1_min = min(m1)
    m1_max = max(m1)

    source = ColumnDataSource(data = dict({(key, tuple([value[i] for i in circleinds if i not in edgeinds])) 
                                           for (key, value) in data.iteritems()}))
    loc = figure(plot_width=450, plot_height=400, toolbar_location="left", x_axis_label='l1 (rad)', y_axis_label='m1 (rad)',
                 x_range=(l1_min, l1_max), y_range=(m1_min,m1_max), tools=tools, webgl=True)
    loc.circle('l1', 'm1', size='sizes', line_color=None, fill_color='colors', fill_alpha=0.2, source=source)

    if crossinds:
        sourceneg = ColumnDataSource(data = dict({(key, tuple([value[i] for i in crossinds]))
                                                  for (key, value) in data.iteritems()}))
        loc.cross('l1', 'm1', size='sizes', line_color='colors', line_alpha=0.2, source=sourceneg)

    if edgeinds:
        sourceedge = ColumnDataSource(data = dict({(key, tuple([value[i] for i in edgeinds]))
                                                   for (key, value) in data.iteritems()}))
        loc.circle('l1', 'm1', size='sizes', line_color='colors', fill_color='colors', source=sourceedge, line_alpha=0.5, fill_alpha=0.2)

    hover = loc.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('SNR', '@snrs'), ('time', '@time'), ('key', '@key')])

    if url_path and fileroot:
        url = '{}/cands_{}_sc@scan-seg@seg-i@candint-dm@dmind-dt@dtind.png'.format(url_path, fileroot)
        taptool = loc.select(type=TapTool)
        taptool.callback = OpenURL(url=url)

    return loc


def plotstat(data, circleinds=None, crossinds=None, edgeinds=None, url_path=None, fileroot=None, tools="hover,tap,pan,box_select,wheel_zoom,reset"):
    """ Make a stat figure """

    if not circleinds: circleinds = range(len(data['snrs']))

    # set ranges
    datalen = len(data['dm'])
    inds = circleinds + crossinds + edgeinds
    specstd = [data['specstd'][i] for i in inds]
    specstd_min = min(specstd)
    specstd_max = max(specstd)
    imkur = [data['imkur'][i] for i in inds]
    imkur_min = min(imkur)
    imkur_max = max(imkur)

    source = ColumnDataSource(data = dict({(key, tuple([value[i] for i in circleinds if i not in edgeinds])) 
                                           for (key, value) in data.iteritems()}))
    stat = figure(plot_width=450, plot_height=400, toolbar_location="left", x_axis_label='Spectral std',
                  y_axis_label='Image kurtosis', x_range=(specstd_min, specstd_max), 
                  y_range=(imkur_min, imkur_max), tools=tools, webgl=True)
    stat.circle('specstd', 'imkur', size='sizes', line_color=None, fill_color='colors', fill_alpha=0.2, source=source)

    if crossinds:
        sourceneg = ColumnDataSource(data = dict({(key, tuple([value[i] for i in crossinds]))
                                                  for (key, value) in data.iteritems()}))
        stat.cross('specstd', 'imkur', size='sizes', line_color='colors', line_alpha=0.2, source=sourceneg)

    if edgeinds:
        sourceedge = ColumnDataSource(data = dict({(key, tuple([value[i] for i in edgeinds]))
                                                   for (key, value) in data.iteritems()}))
        stat.circle('specstd', 'imkur', size='sizes', line_color='colors', fill_color='colors', source=sourceedge, line_alpha=0.5, fill_alpha=0.2)

    hover = stat.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('SNR', '@snrs'), ('time', '@time'), ('key', '@key')])

    if url_path and fileroot:
        url = '{}/cands_{}_sc@scan-seg@seg-i@candint-dm@dmind-dt@dtind.png'.format(url_path, fileroot)
        taptool = stat.select(type=TapTool)
        taptool.callback = OpenURL(url=url)

    return stat


def plotnorm(data, circleinds=[], crossinds=[], edgeinds=[], url_path=None, fileroot=None, tools="hover,tap,pan,box_select,wheel_zoom,reset"):
    """ Make a norm figure """

    if not circleinds: circleinds = range(len(data['snrs']))

    # set ranges
    datalen = len(data['dm'])
    inds = circleinds + crossinds + edgeinds
    specstd = [data['specstd'][i] for i in inds]
    specstd_min = min(specstd)
    specstd_max = max(specstd)
    imkur = [data['imkur'][i] for i in inds]
    imkur_min = min(imkur)
    imkur_max = max(imkur)

    source = ColumnDataSource(data = dict({(key, tuple([value[i] for i in circleinds if i not in edgeinds])) 
                                           for (key, value) in data.iteritems()}))
    norm = figure(plot_width=450, plot_height=400, toolbar_location="left", x_axis_label='SNR observed',
                  y_axis_label='SNR expected', tools=tools, webgl=True)
    norm.circle('snrs', 'zs', size='sizes', line_color=None, fill_color='colors', fill_alpha=0.2, source=source)
#    norm.circle(source.data['snrs'], source.data['zs'], line_color=None, fill_alpha=0.2)

    if crossinds:
        sourceneg = ColumnDataSource(data = dict({(key, tuple([value[i] for i in crossinds]))
                                                  for (key, value) in data.iteritems()}))
        norm.cross('snrs', 'zs', size='sizes', line_color='colors', line_alpha=0.2, source=sourceneg)
#        norm.cross(sourceneg.data['snrs'], sourceneg.data['zs'], line_alpha=0.2)

    if edgeinds:
        sourceedge = ColumnDataSource(data = dict({(key, tuple([value[i] for i in edgeinds]))
                                                   for (key, value) in data.iteritems()}))
        norm.circle('snrs', 'zs', size='sizes', line_color='colors', fill_color='colors', source=sourceedge, line_alpha=0.5, fill_alpha=0.2)
#        norm.circle(sourceedge.data['snrs'], sourceedge.data['zs'], line_alpha=0.5, fill_alpha=0.2)

    hover = norm.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('SNR', '@snrs'), ('time', '@time'), ('key', '@key')])

    if url_path and fileroot:
        url = '{}/cands_{}_sc@scan-seg@seg-i@candint-dm@dmind-dt@dtind.png'.format(url_path, fileroot)
        taptool = norm.select(type=TapTool)
        taptool.callback = OpenURL(url=url)

    return norm


def readdata(mergepkl=None, d=None, cands=None, sizerange=(2,70)):
    """ Converts candidate data to dictionary for bokeh

    Can take merged pkl file or d/cands as read separately.
    """

    # get cands from pkl
    if mergepkl:
        logger.info('Reading {0}'.format(mergepkl))
        loc, prop, d = read_candidates(mergepkl, returnstate=True)
    elif d and cands:
        logger.info('Using provided d/cands')
        loc, prop = cands

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
    key = [tuple(ll) for ll in loc]
    scan = loc[:, scancol]
    seg = loc[:, segmentcol]
    candint = loc[:, 2]
    dmind = loc[:, 3]
    dtind = loc[:, 4]
    beamnum = loc[:, 5]

    logger.info('Setting columns...')
    snrs = prop[:, snrcol]
    abssnr = abs(prop[:, snrcol])
    dm = np.array(d['dmarr'])[loc[:, dmindcol]]
    l1 = prop[:, l1col]
    m1 = prop[:, m1col]
    time = np.array([24*3600*d['segmenttimesdict'][scan[i]][seg[i], 0] + d['inttime']*candint[i] for i in range(len(loc))])
#    time.append(24*3600*d['segmenttimesdict'][k[scancol]][k[segmentcol],0] + d['inttime']*k[intcol])
    specstd = prop[:, specstdcol]
    imkur = prop[:, imkurcol]

    logger.info('Calculating sizes, colors, normprob...')
    time = time - min(time)
    zs = normprob(d, snrs)   # very slow!
    sizes = calcsize(snrs)
    colors = colorsat(l1, m1)

    # if pandas is available use dataframe to allow datashader feature
#    data = DataFrame(data={'snrs': snrs, 'dm': dm, 'l1': l1, 'm1': m1, 'time': time, 'specstd': specstd,
#                           'imkur': imkur, 'scan': scan, 'seg': seg, 'candint': candint, 'dmind': dmind,
#                           'dtind': dtind, 'sizes': sizes, 'colors': colors, 'key': key, 'zs': zs, 'abssnr': abssnr})
#    logger.info('Returning a pandas dataframe')
    data = dict(snrs=snrs, dm=dm, l1=l1, m1=m1, time=time, specstd=specstd,
                imkur=imkur, scan=scan, seg=seg, candint=candint, dmind=dmind,
                dtind=dtind, sizes=sizes, colors=colors, key=key, zs=zs, abssnr=abssnr)

    return data


def findhight(data, ignoret=None, threshold=20):
    """ Find bad time ranges from distribution of candidates.

    ignoret is list of tuples [(t0, t1), (t2, t3)] defining ranges to ignore.
    threshold is made above std of candidate distribution in time.
    Returns the time (in seconds) and counts for bins above threshold.
    """

    time = np.sort(data['time'])

    ww = np.ones(len(time), dtype=bool)  # initialize pass filter
    if ignoret:
        for (t0, t1) in ignoret:
            ww = ww & np.where( (time < t0) | (time > t1), True, False )

    bins = np.round(time[ww]).astype('int')
    counts = np.bincount(bins)
    high = np.where(counts > np.median(counts) + threshold*counts.std())[0]

    return high, counts[high]


def calcinds(data, threshold, ignoret=None):
    """ Find indexes for data above (or below) given threshold. """

    # select by time, too

    inds = []
    for i in range(len(data['time'])):
        snr = data['snrs'][i]
        time = data['time'][i]
        if (threshold >= 0 and snr > threshold):
            if ignoret:
                incl = [t0 for (t0, t1) in ignoret if np.round(time).astype(int) in range(t0,t1)]
                logger.debug('{} {} {} {}'.format(np.round(time).astype(int), t0, t1, incl))
                if not incl:
                    inds.append(i)
            else:
                inds.append(i)
        elif threshold < 0 and snr < threshold:
            if ignoret:
                incl = [t0 for (t0, t1) in ignoret if np.round(time).astype(int) in range(t0,t1)]
                logger.debug('{} {} {} {}'.format(np.round(time).astype(int), t0, t1, incl))
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
    imnoise = np.sort(noises[4])
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
    Z = lambda quan: np.sqrt(2)*erfinv( 2*quan - 1) 
    quan = lambda ntrials, i: (ntrials + 1/2. - i)/ntrials

    # calc number of trials
    npix = d['npixx']*d['npixy']
    if d.has_key('goodintcount'):
        nints = d['goodintcount']
    else:
        nints = d['nints']
    ndms = len(d['dmarr'])
    dtfactor = np.sum([1./i for i in d['dtarr']])    # assumes dedisperse-all algorithm
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


def calcsize(values, sizerange=(2,70), inds=None, plaw=3):
    """ Use set of values to calculate symbol size.

    valus is a list of floats for candidate significance.
    inds is an optional list of indexes to use to calculate symbol size.
    Scaling of symbol size min max set by sizerange tuple (min, max).
    plaw is powerlaw scaling of symbol size from values
    """

    if inds:
        smax = max([abs(values[i]) for i in inds])
        smin = min([abs(values[i]) for i in inds])
    else:
        smax = max([abs(val) for val in values])
        smin = min([abs(val) for val in values])
    return [sizerange[0] + sizerange[1] * ((abs(val) - smin)/(smax - smin))**plaw for val in values]

    
def colorsat(l,m):
    """ Returns color for given l,m
    Designed to look like a color wheel that is more saturated in middle.
    """

    lm = np.zeros(len(l), dtype='complex')
    lm.real = l
    lm.imag = m
    red = 0.5*(1+np.cos(np.angle(lm)))
    green = 0.5*(1+np.cos(np.angle(lm) + 2*3.14/3))
    blue = 0.5*(1+np.cos(np.angle(lm) - 2*3.14/3))
    amp = 256*np.abs(lm)/np.abs(lm).max()
    return ["#%02x%02x%02x" % (np.floor(amp[i]*red[i]), np.floor(amp[i]*green[i]), np.floor(amp[i]*blue[i])) for i in range(len(l))]
