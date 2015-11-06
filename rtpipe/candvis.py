import logging
import numpy as n
from bokeh.plotting import ColumnDataSource, figure, save, output_file, vplot
from bokeh.models import HoverTool
from collections import OrderedDict 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def plot_interactive(mergepkl, thresh=6.8, snrbase=5.5):
    """ Make interactive summary plot with bokeh
    """

    with open(mergepkl,'r') as pkl:
        d = pickle.load(pkl)
        cands = pickle.load(pkl)

    TOOLS = "hover,pan,box_select,wheel_zoom"
    output_file(mergepkl.rstrip('.pkl') + '.html')

    # get columns
    snrorig = [cands[k][0] for k in cands.iterkeys()]
    
    snr,dm,l1,m1,time,specstd,imkur,key,sizes,colors = candfilter(d, cands, 0)
    dm_min = min(dm); dm_max = max(dm)
    time_min = min(time); time_max = max(time)
    specstd_min = min(specstd); specstd_max = max(specstd)
    imkur_min = min(imkur); imkur_max = max(imkur)
    l1_min = min(l1); l1_max = max(l1)
    m1_min = min(m1); m1_max = max(m1)

    snr,dm,l1,m1,time,specstd,imkur,key,sizes,colors = candfilter(d, cands, thresh)
    source = ColumnDataSource(data=dict(snr=snr, dm=dm, l1=l1, m1=m1, time=time, specstd=specstd, imkur=imkur, key=key, sizes=sizes, colors=colors))
    snr,dm,l1,m1,time,specstd,imkur,key,sizes,colors = candfilter(d, cands, -1*thresh)
    sourceneg = ColumnDataSource(data=dict(snr=snr, dm=dm, l1=l1, m1=m1, time=time, specstd=specstd, imkur=imkur, key=key, sizes=sizes, colors=colors))

    # DM-time plot
    dmt = figure(plot_width=900, plot_height=400, toolbar_location="left", x_axis_label='Time (s; rough)', y_axis_label='DM (pc/cm3)', x_range=(time_min, time_max), y_range=(dm_min, dm_max), webgl=True, tools=TOOLS)
    dmt.circle('time', 'dm', size='sizes', source=source, line_color=None, fill_color='colors', fill_alpha=0.3)
    dmt.cross('time', 'dm', size=10, source=sourceneg, line_color='colors', line_alpha=0.3)

    loc = figure(plot_width=450, plot_height=400, toolbar_location="left", x_axis_label='l1 (rad)', y_axis_label='m1 (rad)', x_range=(l1_min, l1_max), y_range=(m1_min,m1_max), tools=TOOLS, webgl=True)
    loc.circle('l1', 'm1', size='sizes', source=source, line_color=None, fill_color='colors', fill_alpha=0.3)
    loc.cross('l1', 'm1', size=10, source=sourceneg, line_color='colors', line_alpha=0.3)

    stat = figure(plot_width=450, plot_height=400, toolbar_location="left", x_axis_label='Spectral std', y_axis_label='Image kurtosis', x_range=(specstd_min, specstd_max), y_range=(imkur_min, imkur_max), tools=TOOLS, webgl=True)
    stat.circle('specstd', 'imkur', size='sizes', source=source, line_color=None, fill_color='colors', fill_alpha=0.3)
    stat.cross('specstd', 'imkur', size=10, source=sourceneg, line_color='colors', line_alpha=0.3)
    
    hover = dmt.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('key', '@key'), ('time', '@time')])
    hover = loc.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('key', '@key'), ('time', '@time')])
    hover = stat.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([('key', '@key'), ('time', '@time')])

    save(vplot(dmt, loc,stat))

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
        sizes = [(s - snrbase)**4 for s in snr]
    elif thresh < 0:
        snr = [cands[k][0] for k in cands.iterkeys() if cands[k][0] < thresh]
        dm = [d['dmarr'][k[3]] for k in cands.iterkeys() if cands[k][0] < thresh]
        l1 = [cands[k][2] for k in cands.iterkeys() if cands[k][0] < thresh]
        m1 = [cands[k][3] for k in cands.iterkeys() if cands[k][0] < thresh]
        time = [d['inttime'] * (k[0] * d['nints'] + k[1] * d['readints'] + k[2]) for k in cands.iterkeys() if cands[k][0] < thresh]
        specstd = [cands[k][4] for k in cands.iterkeys() if cands[k][0] < thresh]
        imkur = [cands[k][8] for k in cands.iterkeys() if cands[k][0] < thresh]
        key = [k for k in cands.iterkeys() if cands[k][0] < thresh]
        sizes = [(s - snrbase)**4 for s in snr]
    else:
        snr = [cands[k][0] for k in cands.iterkeys()]
        dm = [d['dmarr'][k[3]] for k in cands.iterkeys()]
        l1 = [cands[k][2] for k in cands.iterkeys()]
        m1 = [cands[k][3] for k in cands.iterkeys()]
        time = [d['inttime'] * (k[0] * d['nints'] + k[1] * d['readints'] + k[2]) for k in cands.iterkeys()]
        specstd = [cands[k][4] for k in cands.iterkeys()]
        imkur = [cands[k][8] for k in cands.iterkeys()]
        key = [k for k in cands.iterkeys()]
        sizes = [(s - snrbase)**4 for s in snr]

    colors = colorsat(l1,m1)
    return snr,dm,l1,m1,time,specstd,imkur,key,sizes,colors

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
