import logging, pickle, os, glob
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
from scipy.special import erfinv
from bokeh.plotting import ColumnDataSource, Figure, save, output_file, show
from bokeh.models import HoverTool, TapTool, OpenURL
from bokeh.models.layouts import HBox, VBox
from collections import OrderedDict 
from rtpipe.parsecands import read_noise, read_candidates
from time import asctime
import activegit, sys

def initializenb():
    """ Find input files and log initialization info """

    print('Working directory: {0}'.format(os.getcwd()))
    print('Run on {0}'.format(asctime()))
    try:
        fileroot = os.environ['fileroot']
        print('Setting fileroot to {0} from environment variable.\n'.format(fileroot))
        candsfile = 'cands_{0}_merge.pkl'.format(fileroot)
        noisefile = 'noise_{0}_merge.pkl'.format(fileroot)
    except KeyError:
        sdmdir = os.getcwd()
        print('Setting sdmdir to current directory {0}\n'.format(os.path.abspath(sdmdir)))
        candsfiles = glob.glob('cands_*_merge.pkl')
        noisefiles = glob.glob('noise_*_merge.pkl')
        if len(candsfiles) == 1 and len(noisefiles) == 1:
            print('Found one cands/merge file set')
        else:
            print('Found multiple cands/noise file sets. Taking first.')

        candsfile = candsfiles[0]
        noisefile = noisefiles[0]
        fileroot = candsfile.rstrip('_merge.pkl').lstrip('cands_')
    print('Set: \n\t candsfile {} \n\t noisefile {} \n\t fileroot {} '.format(candsfile, noisefile, fileroot))
    return (candsfile, noisefile, fileroot)


def plot_interactive(mergepkl, noisepkl=None, thresh=6.0, thresh_link=7.0, ignoret=None, savehtml=True, url_path='plots'):
    """ Backwards compatible function for making interactive candidate summary plot """

    data = readdata(mergepkl)
    circleinds = calcinds(data, thresh, ignoret)
    crossinds = calcinds(data, -1*thresh, ignoret)
    edgeinds = calcinds(data, thresh_link, ignoret)

    workdir = os.path.dirname(mergepkl)
    fileroot = os.path.basename(mergepkl).rstrip('_merge.pkl').lstrip('cands_')

    logger.info('Total on target time: {} s'.format(calcontime(data, inds=circleinds+crossinds+edgeinds)))

    if noisepkl:
        noiseplot = plotnoisecum(noisepkl)
    else:
        noiseplot = None

    combined = plotall(data, circleinds=circleinds, crossinds=crossinds, edgeinds=edgeinds,
                       htmlname=None, noiseplot=noiseplot, url_path=url_path, fileroot=fileroot)

    if savehtml:
        output_file(mergepkl.rstrip('.pkl') + '.html')
        save(combined)
    else:
        return combined


def plotall(data, circleinds=[], crossinds=[], edgeinds=[], htmlname=None, noiseplot=None, url_path='plots', fileroot=None):
    """ Create interactive plot (preserving links between panels) from data dictionary

    data has keys of snr, time, dm, sizes, key and more.
    Optional index arguments are used to filter full data set.
    This can be used to remove bad segments or apply different symbols to subsets.
    url_path is path difference to png files for taptool. ('../plots' for jupyter notebook, 'plots' for public page)
    fileroot is the sdm file name used as root for all png files.
    """

    # set up data dictionary
    if not circleinds: circleinds = calcinds(data, np.abs(data['snrs']).min())
    if not crossinds: crossinds = calcinds(data, -1*np.abs(data['snrs']).min())

    TOOLS = "hover,tap,pan,box_select,wheel_zoom,reset"

    # set ranges
    datalen = len(data['dm'])
    inds = circleinds + crossinds + edgeinds
    dm = [data['dm'][i] for i in inds]
    dm_min = min(min(dm), max(dm)/1.2)
    dm_max = max(max(dm), min(dm)*1.2)
    time = [data['time'][i] for i in inds]
    time_min = min(time)
    time_max = max(time)
    l1 = [data['l1'][i] for i in inds]
    l1_min = min(l1)
    l1_max = max(l1)
    m1 = [data['m1'][i] for i in inds]
    m1_min = min(m1)
    m1_max = max(m1)
    specstd = [data['specstd'][i] for i in inds]
    specstd_min = min(specstd)
    specstd_max = max(specstd)
    imkur = [data['imkur'][i] for i in inds]
    imkur_min = min(imkur)
    imkur_max = max(imkur)

    # create figures
    dmt = Figure(plot_width=950, plot_height=500, toolbar_location="left", x_axis_label='Time (s; relative)',
                 y_axis_label='DM (pc/cm3)', x_range=(time_min, time_max), y_range=(dm_min, dm_max), 
                 webgl=True, tools=TOOLS)
    loc = Figure(plot_width=450, plot_height=400, toolbar_location="left", x_axis_label='l1 (rad)', y_axis_label='m1 (rad)',
                 x_range=(l1_min, l1_max), y_range=(m1_min,m1_max), tools=TOOLS, webgl=True)
    stat = Figure(plot_width=450, plot_height=400, toolbar_location="left", x_axis_label='Spectral std',
                  y_axis_label='Image kurtosis', x_range=(specstd_min, specstd_max), 
                  y_range=(imkur_min, imkur_max), tools=TOOLS, webgl=True)
    norm = Figure(plot_width=450, plot_height=400, toolbar_location="left", x_axis_label='SNR observed',
                  y_axis_label='SNR expected', tools=TOOLS, webgl=True)

    # create positive symbol source and add glyphs
    source = ColumnDataSource(data = dict({(key, tuple([value[i] for i in circleinds if i not in edgeinds])) 
                                           for (key, value) in data.iteritems()}))
    dmt.circle('time', 'dm', size='sizes', fill_color='colors', line_color=None, fill_alpha=0.2, source=source)
    loc.circle('l1', 'm1', size='sizes', line_color=None, fill_color='colors', fill_alpha=0.2, source=source)
    stat.circle('specstd', 'imkur', size='sizes', line_color=None, fill_color='colors', fill_alpha=0.2, source=source)
    norm.circle('abssnr', 'zs', size='sizes', line_color=None, fill_color='colors', fill_alpha=0.2, source=source)

    # create negative symbol source and add glyphs
    if crossinds:
        sourceneg = ColumnDataSource(data = dict({(key, tuple([value[i] for i in crossinds]))
                                                  for (key, value) in data.iteritems()}))
        dmt.cross('time', 'dm', size='sizes', fill_color='colors', line_alpha=0.3, source=sourceneg)
        loc.cross('l1', 'm1', size='sizes', line_color='colors', line_alpha=0.3, source=sourceneg)
        stat.cross('specstd', 'imkur', size='sizes', line_color='colors', line_alpha=0.3, source=sourceneg)
        norm.cross('abssnr', 'zs', size='sizes', line_color='colors', line_alpha=0.3, source=sourceneg)

    # create linked symbol source and add glyphs
    if edgeinds:
        sourceedge = ColumnDataSource(data = dict({(key, tuple([value[i] for i in edgeinds]))
                                                   for (key, value) in data.iteritems()}))
        dmt.circle('time', 'dm', size='sizes', line_color='colors', fill_color='colors', line_alpha=0.5, fill_alpha=0.2, source=sourceedge)
        loc.circle('l1', 'm1', size='sizes', line_color='colors', fill_color='colors', source=sourceedge, line_alpha=0.5, fill_alpha=0.2)
        stat.circle('specstd', 'imkur', size='sizes', line_color='colors', fill_color='colors', source=sourceedge, line_alpha=0.5, fill_alpha=0.2)
        norm.circle('abssnr', 'zs', size='sizes', line_color='colors', fill_color='colors', source=sourceedge, line_alpha=0.5, fill_alpha=0.2)

    hover = dmt.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('SNR', '@snrs'), ('key', '@key')])
    hover = loc.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('SNR', '@snrs'), ('key', '@key')])
    hover = stat.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('SNR', '@snrs'), ('key', '@key')])
    hover = norm.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('SNR', '@snrs'), ('key', '@key')])

    if url_path and fileroot:
        url = '{}/cands_{}_@key.png'.format(url_path, fileroot)
        taptool = dmt.select(type=TapTool)
        taptool.callback = OpenURL(url=url)
        taptool = loc.select(type=TapTool)
        taptool.callback = OpenURL(url=url)
        taptool = stat.select(type=TapTool)
        taptool.callback = OpenURL(url=url)
        taptool = norm.select(type=TapTool)
        taptool.callback = OpenURL(url=url)

# this approach does not preserve links between panels
#    dmt = plotdmt(data, circleinds=circleinds, crossinds=crossinds, edgeinds=edgeinds, url_path=url_path, fileroot=fileroot, tools=TOOLS) # maybe add size?
#    loc = plotloc(data, circleinds=circleinds, crossinds=crossinds, edgeinds=edgeinds, url_path=url_path, fileroot=fileroot, tools=TOOLS)
#    stat = plotstat(data, circleinds=circleinds, crossinds=crossinds, edgeinds=edgeinds, url_path=url_path, fileroot=fileroot, tools=TOOLS)
#    norm = plotnorm(data, circleinds=circleinds, crossinds=crossinds, edgeinds=edgeinds, url_path=url_path, fileroot=fileroot, tools=TOOLS)

    # arrange figures
    top = HBox(dmt, width=950)
    middle = HBox(loc, stat, width=950)
    if noiseplot:
        bottom = HBox(norm, noiseplot, width=950)
    else:
        bottom = HBox(norm, width=950)
    combined = VBox(top, middle, bottom, width=950)

    if htmlname:
        output_file(htmlname)
        save(combined)
    else:
        return combined


def plotdmt(data, circleinds=[], crossinds=[], edgeinds=[], url_path=None, fileroot=None,
            tools="hover,tap,pan,box_select,wheel_zoom,reset", plot_width=950, plot_height=500):
    """ Make a light-weight dm-time figure """

    fields = ['dm', 'time', 'sizes', 'colors', 'snrs', 'key']

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
                                           for (key, value) in data.iteritems() if key in fields}))
    dmt = Figure(plot_width=plot_width, plot_height=plot_height, toolbar_location="left", x_axis_label='Time (s; relative)',
                 y_axis_label='DM (pc/cm3)', x_range=(time_min, time_max), y_range=(dm_min, dm_max), 
                 webgl=True, tools=tools)
    dmt.circle('time', 'dm', size='sizes', fill_color='colors', line_color=None, fill_alpha=0.2, source=source)

    if crossinds:
        sourceneg = ColumnDataSource(data = dict({(key, tuple([value[i] for i in crossinds]))
                                                  for (key, value) in data.iteritems() if key in fields}))
        dmt.cross('time', 'dm', size='sizes', fill_color='colors', line_alpha=0.3, source=sourceneg)

    if edgeinds:
        sourceedge = ColumnDataSource(data = dict({(key, tuple([value[i] for i in edgeinds]))
                                                   for (key, value) in data.iteritems() if key in fields}))
        dmt.circle('time', 'dm', size='sizes', line_color='colors', fill_color='colors', line_alpha=0.5, fill_alpha=0.2, source=sourceedge)
    hover = dmt.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('SNR', '@snrs'), ('key', '@key')])

    if url_path and fileroot:
#        url = '{}/cands_{}_sc@scan-seg@seg-i@candint-dm@dmind-dt@dtind.png'.format(url_path, fileroot)
        url = '{}/cands_{}_@key.png'.format(url_path, fileroot)
        taptool = dmt.select(type=TapTool)
        taptool.callback = OpenURL(url=url)

    return dmt


def plotloc(data, circleinds=[], crossinds=[], edgeinds=[], url_path=None, fileroot=None,
            tools="hover,tap,pan,box_select,wheel_zoom,reset", plot_width=450, plot_height=400):
    """ Make a light-weight loc figure """

    fields = ['l1', 'm1', 'sizes', 'colors', 'snrs', 'key']

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
                                           for (key, value) in data.iteritems() if key in fields}))
    loc = Figure(plot_width=plot_width, plot_height=plot_height, toolbar_location="left", x_axis_label='l1 (rad)', y_axis_label='m1 (rad)',
                 x_range=(l1_min, l1_max), y_range=(m1_min,m1_max), tools=tools, webgl=True)
    loc.circle('l1', 'm1', size='sizes', line_color=None, fill_color='colors', fill_alpha=0.2, source=source)

    if crossinds:
        sourceneg = ColumnDataSource(data = dict({(key, tuple([value[i] for i in crossinds]))
                                                  for (key, value) in data.iteritems() if key in fields}))
        loc.cross('l1', 'm1', size='sizes', line_color='colors', line_alpha=0.3, source=sourceneg)

    if edgeinds:
        sourceedge = ColumnDataSource(data = dict({(key, tuple([value[i] for i in edgeinds]))
                                                   for (key, value) in data.iteritems() if key in fields}))
        loc.circle('l1', 'm1', size='sizes', line_color='colors', fill_color='colors', source=sourceedge, line_alpha=0.5, fill_alpha=0.2)

    hover = loc.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('SNR', '@snrs'), ('key', '@key')])

    if url_path and fileroot:
        url = '{}/cands_{}_@key.png'.format(url_path, fileroot)
        taptool = loc.select(type=TapTool)
        taptool.callback = OpenURL(url=url)

    return loc


def plotstat(data, circleinds=None, crossinds=None, edgeinds=None, url_path=None, fileroot=None, 
             tools="hover,tap,pan,box_select,wheel_zoom,reset", plot_width=450, plot_height=400):
    """ Make a light-weight stat figure """

    fields = ['imkur', 'specstd', 'sizes', 'colors', 'snrs', 'key']

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
                                           for (key, value) in data.iteritems() if key in fields}))
    stat = Figure(plot_width=plot_width, plot_height=plot_height, toolbar_location="left", x_axis_label='Spectral std',
                  y_axis_label='Image kurtosis', x_range=(specstd_min, specstd_max), 
                  y_range=(imkur_min, imkur_max), tools=tools, webgl=True)
    stat.circle('specstd', 'imkur', size='sizes', line_color=None, fill_color='colors', fill_alpha=0.2, source=source)

    if crossinds:
        sourceneg = ColumnDataSource(data = dict({(key, tuple([value[i] for i in crossinds]))
                                                  for (key, value) in data.iteritems() if key in fields}))
        stat.cross('specstd', 'imkur', size='sizes', line_color='colors', line_alpha=0.3, source=sourceneg)

    if edgeinds:
        sourceedge = ColumnDataSource(data = dict({(key, tuple([value[i] for i in edgeinds]))
                                                   for (key, value) in data.iteritems() if key in fields}))
        stat.circle('specstd', 'imkur', size='sizes', line_color='colors', fill_color='colors', source=sourceedge, line_alpha=0.5, fill_alpha=0.2)

    hover = stat.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('SNR', '@snrs'), ('key', '@key')])

    if url_path and fileroot:
        url = '{}/cands_{}_@key.png'.format(url_path, fileroot)
        taptool = stat.select(type=TapTool)
        taptool.callback = OpenURL(url=url)

    return stat


def plotnorm(data, circleinds=[], crossinds=[], edgeinds=[], url_path=None, fileroot=None,
             tools="hover,tap,pan,box_select,wheel_zoom,reset", plot_width=450, plot_height=400):
    """ Make a light-weight norm figure """

    fields = ['zs', 'sizes', 'colors', 'abssnr', 'key', 'snrs']

    if not circleinds: circleinds = range(len(data['snrs']))

    source = ColumnDataSource(data = dict({(key, tuple([value[i] for i in circleinds if i not in edgeinds])) 
                                           for (key, value) in data.iteritems() if key in fields}))
    norm = Figure(plot_width=plot_width, plot_height=plot_height, toolbar_location="left", x_axis_label='SNR observed',
                  y_axis_label='SNR expected', tools=tools, webgl=True)
    norm.circle('abssnr', 'zs', size='sizes', line_color=None, fill_color='colors', fill_alpha=0.2, source=source)

    if crossinds:
        sourceneg = ColumnDataSource(data = dict({(key, tuple([value[i] for i in crossinds]))
                                                  for (key, value) in data.iteritems() if key in fields}))
        norm.cross('abssnr', 'zs', size='sizes', line_color='colors', line_alpha=0.3, source=sourceneg)

    if edgeinds:
        sourceedge = ColumnDataSource(data = dict({(key, tuple([value[i] for i in edgeinds]))
                                                   for (key, value) in data.iteritems() if key in fields}))
        norm.circle('abssnr', 'zs', size='sizes', line_color='colors', fill_color='colors', source=sourceedge, line_alpha=0.5, fill_alpha=0.2)

    hover = norm.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('SNR', '@snrs'), ('key', '@key')])

    if url_path and fileroot:
        url = '{}/cands_{}_@key.png'.format(url_path, fileroot)
        taptool = norm.select(type=TapTool)
        taptool.callback = OpenURL(url=url)

    return norm


def plotnoise(noisepkl, mergepkl, plot_width=950, plot_height=400):
    """ Make two panel plot to summary noise analysis with estimated flux scale """

    d = pickle.load(open(mergepkl))
    ndist, imstd, flagfrac = plotnoisedist(noisepkl, plot_width=plot_width/2, plot_height=plot_height)
    fluxscale = calcfluxscale(d, imstd, flagfrac)
    logger.info('Median image noise is {0:.3} Jy.'.format(fluxscale*imstd))
    ncum = plotnoisecum(noisepkl, fluxscale=fluxscale, plot_width=plot_width/2, plot_height=plot_height)

    hndl = show(HBox(ndist, ncum, width=plot_width, height=plot_height))
    return hndl


def plotnoisecum(noisepkl, fluxscale=1, plot_width=450, plot_height=400):
    """ Merged noise pkl converted to interactive cumulative histogram 

    noisepkl is standard noise pickle file.
    fluxscale is scaling applied by gain calibrator. telcal solutions have fluxscale=1.
    """

    # noise histogram
    noises = read_noise(noisepkl)
    imnoise = np.sort(fluxscale*noises[4])
    frac = [float(count)/len(imnoise) for count in reversed(range(1, len(imnoise)+1))]
    noiseplot = Figure(plot_width=plot_width, plot_height=plot_height, toolbar_location="above",
                       x_axis_label='Image noise (Jy; cal scaling {0:.3})'.format(fluxscale),
                       y_axis_label='Cumulative fraction', tools='pan, wheel_zoom, reset')
    noiseplot.line(imnoise, frac)

    return noiseplot


def plotnoisedist(noisepkl, plot_width=450, plot_height=400):
    """ """

    # plot noise and flag distributions
    scans, segments, noiseperbl, flagfrac, imstd = read_noise(noisepkl)
    pl = Figure(plot_width=plot_width, plot_height=plot_height, toolbar_location="above",
                x_axis_label='Flag fraction', y_axis_label='Image noise (sys)', tools='pan, wheel_zoom, reset')
    pl.cross(flagfrac, imstd, line_alpha=0.2, color='blue')

    # find medians
    flagfrac_med = np.median(flagfrac)
    imstd_med = np.median(imstd)
    logger.info('Median image noise (sys) = {0}'.format(imstd_med))
    logger.info('Median flag fraction = {0}'.format(flagfrac_med))
    pl.cross(flagfrac_med, imstd_med, line_alpha=1, size=40, color='red')

    # estimate number of zero noise images
    zeronoisefrac = float(len(np.where(imstd == 0.)[0]))/len(imstd)
    logger.info('{0:.0%} of noise images are zeros'.format(zeronoisefrac))
    
    return pl, imstd_med, flagfrac_med


def calcfluxscale(d, imstd_med, flagfrac_med):
    """ Given state dict and noise properties, estimate flux scale at the VLA 

    imstd and flagfrac are expected to be median (typical) values from sample in merged noise pkl.
    """

    # useful functions and VLA parameters
    sensitivity = lambda sefd, dt, bw, eta, nbl, npol: sefd/(eta*np.sqrt(nbl*2 * dt * bw * npol))
    nbl = lambda nant: nant*(nant-1)/2
    eta = {'L': 0.92, 'S': 0.92, 'C': 0.8, 'X': 0.8}   # correlator efficiency
    sefd = {'L': 420, 'S': 370, 'C': 310, 'X': 250}    # fixed to match exposure calculator int time to 100 microJy.

    bw = sum([d['spw_nchan_select'][i]*d['spw_chansize'][i] for i in range(len(d['spw_chansize']))])
    dt = d['inttime']
    npol = d['npol']
    nant = d['nants']
    freq = d['freq'][0]
    if (freq >= 1 and freq < 2):
        band = 'L'
    elif (freq >= 2 and freq < 4):
        band = 'S'
    elif (freq >= 4 and freq < 8):
        band = 'C'
    elif (freq >= 8 and freq < 12):
        band = 'X'
    else:
        logger.warn('first channel freq ({0}) not in bands L, S, C, or X. Assuming L band.'.format(freq))
        band = 'L'

    goodfrac = 1 - flagfrac_med # correct for flagged data
    slim_theory = sensitivity(sefd[band], dt, bw, eta[band], goodfrac*nbl(nant), npol)
    fluxscale = slim_theory/imstd_med

    return fluxscale


def readdata(mergepkl=None, d=None, cands=None, sizerange=(2,70)):
    """ Converts candidate data to dictionary for bokeh

    Can take merged pkl file or d/cands as read separately.
    cands is an optional (loc, prop) tuple of numpy arrays.
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
    key = ['sc{0}-seg{1}-i{2}-dm{3}-dt{4}'.format(ll[scancol], ll[segmentcol], ll[intcol], ll[dmindcol], ll[dtindcol]) for ll in loc]
#    key = [tuple(ll) for ll in loc]
    scan = loc[:, scancol]
    seg = loc[:, segmentcol]
    candint = loc[:, 2]
    dmind = loc[:, 3]
    dtind = loc[:, 4]
    beamnum = loc[:, 5]

    logger.info('Setting columns...')
    snrs = prop[:, snrcol]
    abssnr = np.abs(prop[:, snrcol])
    dm = np.array(d['dmarr'])[loc[:, dmindcol]]
    l1 = prop[:, l1col]
    m1 = prop[:, m1col]
    time = np.array([24*3600*d['segmenttimesdict'][scan[i]][seg[i], 0] + d['inttime']*candint[i] for i in range(len(loc))])
#    time.append(24*3600*d['segmenttimesdict'][k[scancol]][k[segmentcol],0] + d['inttime']*k[intcol])
    specstd = prop[:, specstdcol]
    imkur = prop[:, imkurcol]

    logger.info('Calculating sizes, colors, normprob...')
    time = time - min(time)
    sizes = calcsize(snrs)
    colors = colorsat(l1, m1)
    zs = normprob(d, snrs)

    # if pandas is available use dataframe to allow datashader feature
#    data = DataFrame(data={'snrs': snrs, 'dm': dm, 'l1': l1, 'm1': m1, 'time': time, 'specstd': specstd,
#                           'imkur': imkur, 'scan': scan, 'seg': seg, 'candint': candint, 'dmind': dmind,
#                           'dtind': dtind, 'sizes': sizes, 'colors': colors, 'key': key, 'zs': zs, 'abssnr': abssnr})
#    logger.info('Returning a pandas dataframe')
    data = dict(snrs=snrs, dm=dm, l1=l1, m1=m1, time=time, specstd=specstd, scan=scan,
                imkur=imkur, sizes=sizes, colors=colors, key=key, zs=zs, abssnr=abssnr)
#                dtind=dtind, scan=scan, seg=seg, candint=candint, dmind=dmind,

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


def normprob(d, snrs, inds=None, version=2):
    """ Uses observed SNR distribution to calculate normal probability SNR

    Uses state dict to calculate number of trials.
    snrs is list of all snrs in distribution.
    version used to toggle for tests. version 2 is fastest and returns zeros for filtered snr values.
    Returns list of expected snr given each input value's frequency of occurrence via the normal probability assumption
    """

    if not inds: inds = range(len(snrs))

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
    if version == 2:
        # purely sort and numpy-based
        sortinds = np.argsort(snrs[inds])
        lenpos = len(np.where(snrs[inds] >= 0)[0])
        lenneg = len(np.where(snrs[inds] < 0)[0])
        unsortinds = np.zeros(len(sortinds), dtype=int)
        unsortinds[sortinds] = np.arange(len(sortinds))
        rank = np.concatenate( (np.arange(1, lenneg+1), np.arange(1, lenpos+1)[::-1]) )
        logger.debug('{} {}'.format(rank, sortinds))
        zval = Z(quan(ntrials, rank[unsortinds]))
        if inds != range(len(snrs)):  # add zeros for filtered data to match length to original snr array
            zval = np.array([zval[inds.index(i)] if i in inds else 0 for i in range(len(snrs))])

    elif version == 1:
        # numpy array based
        snrpos = snrs[inds][np.where(snrs[inds] > 0)]
        snrneg = snrs[inds][np.where(snrs[inds] < 0)]
        snrsortpos = np.sort(snrpos)[::-1]
        snrsortneg = np.sort(snrneg)
        
        logger.debug('Sorted pos/neg SNRs')

        zval = []
        for i,snr in enumerate(snrs):
            if i in inds:
                if snr in snrsortpos:
                    zval.append(Z(quan(ntrials, np.where(snr == snrsortpos)[0][0]+1)))
                elif snr in snrsortneg:
                    zval.append(Z(quan(ntrials, np.where(snr == snrsortneg)[0][0]+1)))

    elif version == 0:
        # list based
        snrsortpos = []
        snrsortneg = []
        for i in inds:
            if snrs[i] > 0:
                snrsortpos.append(snrs[i])
            elif snrs[i] < 0:
                snrsortneg.append(abs(snrs[i]))

        snrsortpos = sorted(snrsortpos, reverse=True)
        snrsortneg = sorted(snrsortneg, reverse=True)
        logger.debug('Sorted pos/neg SNRs')

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

    values is a list of floats for candidate significance.
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


def filterdata(data, plinds, d, threshold, ignorestr, thresh0=6., thresh1=7.):
    """ Iteratively filter bad times and set indices for later plotting """

    ignoret = parseignoret(ignorestr)
    plinds['cir'] = calcinds(data, thresh0, ignoret=ignoret) # positive cands
    plinds['cro'] = calcinds(data, -1*thresh0, ignoret=ignoret) # negative cands
    plinds['edg'] = calcinds(data, thresh1, ignoret=ignoret) # cands with png plots
    sortinds = sorted(set(plinds['cir'] + plinds['cro'] + plinds['edg']))
    print('Selected {} ({} linked) points.'.format(len(sortinds), len(plinds['edg'])))
    
    print('Estimated total on target time: {} s\n'.format(calcontime(
        data, inds=plinds['cir']+plinds['cro']+plinds['edg'])))
    
    # these must get get rescaled when cands are ignored
    data['zs'] = normprob(d, data['snrs'], inds=sortinds)   

    # print high 1s bin counts
    hight, highcount = findhight(data, ignoret=ignoret, threshold=threshold)
    if len(hight):
        print('High times \t High counts:')
        for i in range(len(hight)):
              print('{0}\t{1}'.format(hight[i], highcount[i]))
    else:
        print('No high 1s bin counts.')
    print('\n')

    # print high cands and their times
    biginds = np.argsort(data['abssnr'][sortinds])[-5:]    
    print('Top 5 abs(snr) candidates and times:')
    for ind in biginds[::-1]:
        print(data['snrs'][sortinds][ind], data['time'][sortinds][ind])
    print('\n')


def parseignoret(ignorestr):
    if ',' in ignorestr:
        ignorelist = ignorestr.split(',')
        assert (len(ignorelist)/2.).is_integer(), 'ignorestr be pairs of comma-delimited values.'
        ignoret = [(int(ignorelist[i]), int(ignorelist[i+1])) for i in range(0, len(ignorelist), 2)]
    else:
        ignoret = []
    return ignoret        


def displayplot(data, plinds, plottype, scaling, fileroot, url_path='http://www.aoc.nrao.edu/~claw/plots'):
    """ Generate interactive plot """

    plotdict = {'dmt': plotdmt, 'norm': plotnorm,
               'loc': plotloc, 'stat': plotstat,
               'all': plotall}
    sizedict = {'dmt': [900,500], 'norm': [700, 700], 'loc': [700,700],
                'stat': [700,700]}

    sortinds = sorted(set(plinds['cir'] + plinds['cro'] + plinds['edg']))
    sizesrc, plaw = scaling.split('_')
    data['sizes'] = calcsize(data[sizesrc], inds=sortinds, plaw=int(plaw))

    if plottype != 'all':
        wid, hei = sizedict[plottype]
        pl = plotdict[plottype](data, circleinds=plinds['cir'], crossinds=plinds['cro'],
                                edgeinds=plinds['edg'], url_path=url_path,
                                fileroot=fileroot, plot_width=wid, plot_height=hei)
    else:
        pl = plotall(data, circleinds=plinds['cir'], crossinds=plinds['cro'],
                                 edgeinds=plinds['edg'], url_path=url_path,
                                 fileroot=fileroot)
    hdl = show(pl)


def addclassifications(agdir, prop, version=None, statfeats = [0,4,5,6,7,8]):
    """ Calculates real score probability of prop from an activegit repo.

    version is string name of activegit tag.
    Default agdir initialization will have latest tag, so version is optional.
    statfeats set to work with alnotebook naming.
    """

    try:
        ag = activegit.ActiveGit(agdir)
        if version:
            ag.set_version(version)
        clf = ag.read_classifier()

        score = clf.predict_proba((np.nan_to_num(prop[:,statfeats])))[:,1]  # take real score
        return score
    except:
        logger.info('Failure when parsing activegit repo or applying classification.\n{0}'.format(sys.exc_info()[0]))
        return []

