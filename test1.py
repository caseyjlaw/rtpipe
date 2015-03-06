# test script to be run interactively
import realtime.parsesdm as ps
import realtime.RT as rt
import realtime.parsecal as pc
import realtime.rtlib_cython as rtlib
import argparse
import numpy as n
import multiprocessing.sharedctypes as mps

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="filename with full path")
parser.add_argument("scan", help="scan to search.")
parser.add_argument("--gainfile", help="gain calibration table", default='')
parser.add_argument("--bpfile", help="bandpass calibration table", default='')
parser.add_argument("--nints", help="number of segments to split", default=100)
parser.add_argument("--npix", help="number of pixels in image", default=0)
parser.add_argument("--uvres", help="uv resolution for imaging", default=0)
parser.add_argument("--nsegments", help="number of segments", default=0)
parser.add_argument("--segment", help="segment to load", default=0)
parser.add_argument("--dmarr", help="dmarray (comma-delimited)", default='')
parser.add_argument("--ii", help="integration to image", default=0)
args = parser.parse_args(); filename = args.filename; scan = int(args.scan); nints = int(args.nints); ii = int(args.ii)
gainfile = args.gainfile; bpfile = args.bpfile
npix = int(args.npix); uvres = int(args.uvres)
if args.dmarr:
    dmarr = [int(i) for i in args.dmarr.split(',')]
else:
    dmarr = [0]

# parameters
sigma_image1 = 7
timesub = 'mean'
flagmode = 'standard'
nthread = 16
chans = range(6,122)+range(134,250)
#chans = range(3,61)+range(67,125)
searchtype = 'image1'
test = 'pipeline'  # options: 'all', 'pipeline', 'wterm'

global data
 
# set up pipeline
state = ps.get_metadata(filename, scan, chans=chans)
if args.nsegments == 0:
    nsegments = state['nints']/nints
else:
    nsegments = int(args.nsegments)
rt.set_pipeline(state, nthread=nthread, nsegments=nsegments, gainfile=gainfile, bpfile=bpfile, dmarr=dmarr, dtarr=[1], timesub=timesub, candsfile='', noisefile='', sigma_image1=sigma_image1, flagmode=flagmode, flagantsol=True, searchtype=searchtype, uvres=uvres, npix=npix, read_downsample=1)
if args.segment == 0:
    segment = state['nsegments']/2
else:
    segment = int(args.segment)

# read and prep data
if (test == 'all') or (test == 'wterm'):
    t0 = state['segmenttimes'][segment][0]
    t1 = state['segmenttimes'][segment][1]
    readints = n.round(24*3600*(t1 - t0)/state['inttime'], 0).astype(int)/state['read_downsample']
    data_mem = mps.RawArray(mps.ctypes.c_float, long(readints*state['nbl']*state['nchan']*state['npol'])*2)  # 'long' type needed to hold whole (2 min, 5 ms) scans
    data = rt.numpyview(data_mem, 'complex64', (readints, state['nbl'], state['nchan'], state['npol']))
    data[:] = ps.read_bdf_segment(state, segment)

    if state['gainfile']:
        sols = pc.casa_sol(state['gainfile'], flagants=state['flagantsol'])
        sols.parsebp(state['bpfile'])
        sols.setselection(state['segmenttimes'][segment].mean(), state['freq']*1e9)
        sols.apply(data, state['blarr'])

    rt.dataflagall(data, state)

    if state['timesub'] == 'mean':
        rtlib.meantsub(data)

    # dedisperse if nonzero
    if state['dmarr'][0] > 0:
        dt = 1
        rtlib.dedisperse_resample(data, state['freq'], state['inttime'], state['dmarr'][0], dt, verbose=1)        # dedisperses data.

    # do analysis
    (u,v,w) = ps.get_uvw_segment(state, segment)
    image = rt.sample_image(state, data, u, v, w, ii)
    if test =='wterm':
        imagew = rt.sample_image(state, data, u, v, w, ii, imager='w')

elif test == 'pipeline':
    rt.pipeline(state, segment)
