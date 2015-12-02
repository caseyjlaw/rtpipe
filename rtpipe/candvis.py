from scipy.special import erfinv
import numpy as n
import logging, pickle, os
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from bokeh.plotting import ColumnDataSource, figure, save, output_file, vplot
from bokeh.models.widgets import VBox, HBox
from bokeh.models import HoverTool, TapTool, OpenURL
from collections import OrderedDict 
from rtpipe.parsecands import read_noise

def plot_interactive(mergepkl, noisepkl='', thresh=6.0, savehtml=True, urlbase='http://www.aoc.nrao.edu/~claw/realfast/plots'):
    """ Make interactive summary plot with bokeh
    if savehtml will write to html, otherwise returns tuple of bokeh plot objects
    saves to html locally and point to plots in urlbase.
    """

    with open(mergepkl,'r') as pkl:
        d = pickle.load(pkl)
        cands = pickle.load(pkl)

    assert 'scan' in d['featureind'], 'This does not appear to be a merged cands pkl file.'

    # try to find noisepkl
    if not noisepkl:
        noisetest = os.path.join(os.path.dirname(mergepkl), 'noise' + os.path.basename(mergepkl).lstrip('cands'))
        if os.path.exists(noisetest):
            noisepkl = noisetest

    TOOLS = "tap,hover,pan,box_select,wheel_zoom,reset"
    output_file(mergepkl.rstrip('.pkl') + '.html')

    # get columns
    snrorig = [cands[k][0] for k in cands.iterkeys()]
    
    snr,dm,l1,m1,time,specstd,imkur,key,sizes,colors,zs = candfilter(d, cands, thresh=0)
    dm_min = min(min(dm), max(dm)/1.2); dm_max = max(max(dm), min(dm)*1.2)
    time_min = min(time); time_max = max(time)
    specstd_min = min(specstd); specstd_max = max(specstd)
    imkur_min = min(imkur); imkur_max = max(imkur)
    l1_min = min(l1); l1_max = max(l1)
    m1_min = min(m1); m1_max = max(m1)

    snr,dm,l1,m1,time,specstd,imkur,key,sizes,colors,zs = candfilter(d, cands, thresh)
    scan, seg, candint, dmind, dtind, beamnum = zip(*key)  # unpack key into individual columns
    source = ColumnDataSource(data=dict(snr=snr, dm=dm, l1=l1, m1=m1, time=time, specstd=specstd, imkur=imkur, scan=scan, seg=seg, candint=candint, dmind=dmind, dtind=dtind, sizes=sizes, colors=colors, zs=zs, key=key))

    snr,dm,l1,m1,time,specstd,imkur,key,sizes,colors,zs = candfilter(d, cands, -1*thresh)
    scan, seg, candint, dmind, dtind, beamnum = zip(*key)  # unpack key into individual columns
    sourceneg = ColumnDataSource(data=dict(snr=snr, dm=dm, l1=l1, m1=m1, time=time, specstd=specstd, imkur=imkur, scan=scan, seg=seg, candint=candint, dmind=dmind, dtind=dtind, sizes=sizes, colors=colors, zs=zs, abssnr=n.abs(snr), key=key))

    # DM-time plot
    dmt = figure(plot_width=950, plot_height=400, toolbar_location="left", x_axis_label='Time (s; rough)', y_axis_label='DM (pc/cm3)', x_range=(time_min, time_max), y_range=(dm_min, dm_max), webgl=True, tools=TOOLS)
    dmt.circle('time', 'dm', size='sizes', source=source, line_color=None, fill_color='colors', fill_alpha=0.2)
    dmt.cross('time', 'dm', size='sizes', source=sourceneg, line_color='colors', line_alpha=0.2)

    # image location plot
    loc = figure(plot_width=450, plot_height=400, toolbar_location="left", x_axis_label='l1 (rad)', y_axis_label='m1 (rad)', x_range=(l1_min, l1_max), y_range=(m1_min,m1_max), tools=TOOLS, webgl=True)
    loc.circle('l1', 'm1', size='sizes', source=source, line_color=None, fill_color='colors', fill_alpha=0.2)
    loc.cross('l1', 'm1', size='sizes', source=sourceneg, line_color='colors', line_alpha=0.2)

    # cand spectrum/image statistics plot
    stat = figure(plot_width=450, plot_height=400, toolbar_location="left", x_axis_label='Spectral std', y_axis_label='Image kurtosis', x_range=(specstd_min, specstd_max), y_range=(imkur_min, imkur_max), tools=TOOLS, webgl=True)
    stat.circle('specstd', 'imkur', size='sizes', source=source, line_color=None, fill_color='colors', fill_alpha=0.2)
    stat.cross('specstd', 'imkur', size='sizes', source=sourceneg, line_color='colors', line_alpha=0.2)
    
    # norm prob plot
    norm = figure(plot_width=450, plot_height=400, toolbar_location="left", x_axis_label='SNR observed', y_axis_label='SNR expected', tools=TOOLS, webgl=True)
    norm.circle('snr', 'zs', size='sizes', source=source, line_color=None, fill_color='colors', fill_alpha=0.2)
    norm.cross('abssnr', 'zs', size='sizes', source=sourceneg, line_color='colors', line_alpha=0.2)

    # noise histogram
    if noisepkl:
        logger.info('Found merged noise file at %s' % noisepkl)
        noises = read_noise(noisepkl)
        imnoise = n.sort(noises[4])
        frac = [float(count)/len(imnoise) for count in reversed(range(1, len(imnoise)+1))]
        noiseplot = figure(plot_width=450, plot_height=400, toolbar_location="left", x_axis_label='Noise image std', y_axis_label='Cumulative fraction', tools='pan, wheel_zoom, reset')
        noiseplot.line(imnoise, frac)
    else:
        logger.info('No merged noise file at %s' % noisepkl)

    # define hover and url behavior
    hover = dmt.select(dict(type=HoverTool)); hover.tooltips = OrderedDict([('SNR', '@snr'), ('time', '@time'), ('key', '@key')])
    hover = loc.select(dict(type=HoverTool)); hover.tooltips = OrderedDict([('SNR', '@snr'), ('time', '@time'), ('key', '@key')])
    hover = stat.select(dict(type=HoverTool));  hover.tooltips = OrderedDict([('SNR', '@snr'), ('time', '@time'), ('key', '@key')])
    hover = norm.select(dict(type=HoverTool));  hover.tooltips = OrderedDict([('SNR', '@snr'), ('time', '@time'), ('key', '@key')])
    url = '%s/%s_sc@scan-seg@seg-i@candint-dm@dmind-dt@dtind.png' % (urlbase, os.path.basename(mergepkl.rstrip('_merge.pkl')) )
    taptool = dmt.select(type=TapTool);  taptool.callback = OpenURL(url=url)
    taptool = loc.select(type=TapTool);  taptool.callback = OpenURL(url=url)    
    taptool = stat.select(type=TapTool);  taptool.callback = OpenURL(url=url)    
    taptool = norm.select(type=TapTool);  taptool.callback = OpenURL(url=url)

    # arrange plots
    top = HBox(children=[dmt])
    middle = HBox(children=[loc, stat])
    if noisepkl:
        bottom = HBox(children=[norm, noiseplot])
    else:
        bottom = HBox(children=[norm])
    combined = VBox(children=[top,middle,bottom])

    if savehtml:
        save(combined)
    else:
        return combined

def normprob(d, snrs):
    """ Function takes state dict and snr list 
    Returns list of expected snr given each input value's frequency of occurrence via the normal probability assumption
    input should be all of one sign (all pos or neg)
    """

    signs = [n.sign(snr) for snr in snrs]
    assert all(signs) or not all(signs), 'Signs of all snr values should be the same'

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
    try:
        if n.sign(snrs[0]) > 0:
            snrsort = sorted([s for s in snrs if s > 0], reverse=True)
            zval = [Z(quan(ntrials, snrsort.index(snr)+1)) for snr in snrs]   # assumes unique snr values
        elif n.sign(snrs[0]) < 0:
            snrsort = sorted([abs(s) for s in snrs if s < 0], reverse=True)
            zval = [Z(quan(ntrials, snrsort.index(abs(snr))+1)) for snr in snrs if snr < 0]
    except IndexError, ValueError:
        zval = []

    return zval

def candfilter(d, cands, thresh=0):
    """ filters candidate data. if thresh is negative, returns values less than thresh.
    default thresh=0 returns all
    """

    if thresh > 0:
        snr = [cands[k][0] for k in cands.iterkeys() if cands[k][0] > thresh]
        dm = [d['dmarr'][k[3]] for k in cands.iterkeys() if cands[k][0] > thresh]
        l1 = [cands[k][2] for k in cands.iterkeys() if cands[k][0] > thresh]
        m1 = [cands[k][3] for k in cands.iterkeys() if cands[k][0] > thresh]
        time = [d['inttime'] * (k[0] * d['nints'] + k[1] * d['readints'] + k[2]) for k in cands.iterkeys() if cands[k][0] > thresh]
        specstd = [cands[k][4] for k in cands.iterkeys() if cands[k][0] > thresh]
        imkur = [cands[k][8] for k in cands.iterkeys() if cands[k][0] > thresh]
        key = [k for k in cands.iterkeys() if cands[k][0] > thresh]
        zs = normprob(d, snr)
    elif thresh < 0:
        snr = [cands[k][0] for k in cands.iterkeys() if cands[k][0] < thresh]
        dm = [d['dmarr'][k[3]] for k in cands.iterkeys() if cands[k][0] < thresh]
        l1 = [cands[k][2] for k in cands.iterkeys() if cands[k][0] < thresh]
        m1 = [cands[k][3] for k in cands.iterkeys() if cands[k][0] < thresh]
        time = [d['inttime'] * (k[0] * d['nints'] + k[1] * d['readints'] + k[2]) for k in cands.iterkeys() if cands[k][0] < thresh]
        specstd = [cands[k][4] for k in cands.iterkeys() if cands[k][0] < thresh]
        imkur = [cands[k][8] for k in cands.iterkeys() if cands[k][0] < thresh]
        key = [k for k in cands.iterkeys() if cands[k][0] < thresh]
        zs = normprob(d, snr)
    else:
        snr = [cands[k][0] for k in cands.iterkeys()]
        dm = [d['dmarr'][k[3]] for k in cands.iterkeys()]
        l1 = [cands[k][2] for k in cands.iterkeys()]
        m1 = [cands[k][3] for k in cands.iterkeys()]
        time = [d['inttime'] * (k[0] * d['nints'] + k[1] * d['readints'] + k[2]) for k in cands.iterkeys()]
        specstd = [cands[k][4] for k in cands.iterkeys()]
        imkur = [cands[k][8] for k in cands.iterkeys()]
        key = [k for k in cands.iterkeys()]
        zs = []

    sizes = calcsize(snr)
    colors = colorsat(l1, m1)
    return snr, dm, l1, m1, time, specstd, imkur, key, sizes, colors, zs

def calcsize(snr, sizerange=(3,60)):
    """ Takes snr list and returns value to scale symbol size.
    """

    if snr:
        smax = max([abs(s) for s in snr])
        smin = min([abs(s) for s in snr])
        return [sizerange[0] + sizerange[1] * ((abs(s) - smin)/(smax - smin))**3 for s in snr]
    else:
        return []
    
def colorsat(l,m):
    """ Returns color for given l,m
    Designed to look like a color wheel that is more saturated in middle.
    """

    if l:
        lm = n.zeros(len(l), dtype='complex')
        lm.real = l; lm.imag = m
        red = 0.5*(1+n.cos(n.angle(lm)))
        green = 0.5*(1+n.cos(n.angle(lm) + 2*3.14/3))
        blue = 0.5*(1+n.cos(n.angle(lm) - 2*3.14/3))
        amp = 256*n.abs(lm)/n.abs(lm).max()
        return ["#%02x%02x%02x" % (n.floor(amp[i]*red[i]), n.floor(amp[i]*green[i]), n.floor(amp[i]*blue[i])) for i in range(len(l))]
    else:
        return []
