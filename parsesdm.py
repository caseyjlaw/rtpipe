#
# functions to read and convert sdm files
# claw, 14jun03
#

import numpy as n
import os, shutil, subprocess, glob, string
import casautil, tasklib
import sdmreader, sdmpy
import rtpipe.parseparams as pp

qa = casautil.tools.quanta()
me = casautil.tools.measures()

def get_metadata(filename, scan, spw=[], chans=[], params=''):
    """ Parses sdm file to define metadata for observation, including scan info, image grid parameters, pipeline memory usage, etc.
    Mirrors parsems.get_metadata(), though not all metadata set here yet.
    If params defined, it will use it (filename or RT.Params instance ok).
    spw/chans argument here will overload params file definition.
    """

    # create primary state dictionary
    d = {}

    # define parameters of pipeline via Params object
    params = pp.Params(params)
    for k in params.defined:   # fill in default params
        d[k] = params[k]
    if len(chans):
        d['chans'] = chans
    if len(spw):
        d['spw'] = spw

    # get workdir set
    d['workdir'], d['filename'] = os.path.split(filename.rstrip('/'))
    if len(d['workdir']):    # workdir may be cwd
        os.chdir(d['workdir'])

    # define scan list
    scans, sources = sdmreader.read_metadata(d['filename'])
    d['scanlist'] = sorted(scans.keys())

    # define spectral info
    sdm = sdmpy.SDM(d['filename'])
    d['spw_orig'] = [int(row.spectralWindowId.split('_')[1]) for row in sdm['SpectralWindow']]
    d['spw_reffreq'] = [float(row.chanFreqStart) for row in sdm['SpectralWindow']]
    d['spw_nchan'] = [int(row.numChan) for row in sdm['SpectralWindow']]
    d['spw_chansize'] = [float(row.chanFreqStep) for row in sdm['SpectralWindow']]
    if not len(spw):
        spw = range(len(d['spw_orig']))
    d['spw'] = n.sort(d['spw_orig'])[spw]

    spwch = []
    reffreq = d['spw_reffreq']; spectralwindow = d['spw_orig']; numchan = d['spw_nchan']; chansize = d['spw_chansize']
    for freq in sorted(d['spw_reffreq']):
        ii = reffreq.index(freq)
        if spectralwindow[ii] in d['spw']:
            spwch.extend(list(n.linspace(reffreq[ii], reffreq[ii]+(numchan[ii]-1)*chansize[ii], numchan[ii])))  # spacing of channel *centers*
    d['freq_orig'] = n.array(spwch).astype('float32')/1e9
           
    d['nspw'] = len(d['spw'])
    if len(chans):
        d['freq'] = d['freq_orig'][chans]
        d['chans'] = chans
    else:
        d['freq'] = d['freq_orig']
        d['chans'] = range(len(d['freq']))
    d['nchan'] = len(d['freq'])

    # define image params
    d['urange'] = {}; d['vrange'] = {}
    d['scan'] = scan
    (u, v, w) = sdmreader.calc_uvw(d['filename'], d['scan'])  # default uses time at start
    u = u * d['freq_orig'][0] * (1e9/3e8) * (-1)     
    v = v * d['freq_orig'][0] * (1e9/3e8) * (-1)     
    d['urange'][d['scan']] = u.max() - u.min()
    d['vrange'][d['scan']] = v.max() - v.min()
    d['uvres_full'] = n.round(25./(3e-1/d['freq'].max())/2).astype('int')    # full VLA field of view. assumes freq in GHz
    # **this may let vis slip out of bounds. should really define grid out to 2*max(abs(u)) and 2*max(abs(v)). in practice, very few are lost.**

    urange = d['urange'][d['scan']]*(d['freq'].max()/d['freq_orig'][0])   # uvw from get_uvw already in lambda at ch0
    vrange = d['vrange'][d['scan']]*(d['freq'].max()/d['freq_orig'][0])
    powers = n.fromfunction(lambda i,j: 2**i*3**j, (14,10), dtype='int')   # power array for 2**i * 3**j
    rangex = n.round(urange).astype('int')
    rangey = n.round(vrange).astype('int')
    largerx = n.where(powers-rangex/d['uvres_full'] > 0, powers, powers[-1,-1])
    p2x, p3x = n.where(largerx == largerx.min())
    largery = n.where(powers-rangey/d['uvres_full'] > 0, powers, powers[-1,-1])
    p2y, p3y = n.where(largery == largery.min())
    d['npixx_full'] = (2**p2x * 3**p3x)[0]
    d['npixy_full'] = (2**p2y * 3**p3y)[0]

    # define ants/bls
    d['ants'] = [int(ant.name.lstrip('ea')) for ant in sdm['Antenna']]
    d['nants'] = len(d['ants'])
#    d['blarr'] = n.array([[d['ants'][i],d['ants'][j]] for i in range(d['nants'])  for j in range(i+1, d['nants'])])
    d['blarr'] = n.array([[d['ants'][i],d['ants'][j]] for j in range(d['nants']) for i in range(0,j)])
    d['nbl'] = len(d['blarr'])

    # define times
    d['starttime_mjd'] = scans[d['scan']]['startmjd']
    # assume inttime same for all scans
    sdm['Main']

    for scan in sdm['Main']:
        interval = int(scan.interval)
        nints = int(scan.numIntegration)
        inttime = 1e-9*interval/nints
        scannum = int(scan.scanNumber)
        if scannum == d['scan']:
            d['inttime'] = inttime
            d['nints'] = nints

    # define pols
    pols = [pol.corrType for pol in sdm['Polarization'] if pol in ['XX', 'YY', 'XY', 'YX', 'RR', 'LL', 'RL', 'LR']]
    d['npol'] = int(pol.numCorr)
    d['pols'] = pols

    # summarize metadata
    print 'Metadata summary:'
    print '\t Using scan %d' % (int(d['scan']))
    print '\t nants, nbl: %d, %d' % (d['nants'], d['nbl'])
    print '\t Freq range (%.3f -- %.3f). %d spw with %d chans.' % (d['freq'].min(), d['freq'].max(), d['nspw'], d['nchan'])
    print '\t Scan has %d ints (%.1f s) and inttime %.3f s' % (d['nints'], d['nints']*d['inttime'], d['inttime'])
    print '\t %d polarizations: %s' % (d['npol'], d['pols'])
    print '\t Ideal uvgrid npix=(%d,%d) and res=%d' % (d['npixx_full'], d['npixy_full'], d['uvres_full'])

    return d

def read_bdf_segment(d, segment=-1):
    """ Reads bdf (sdm) format data into numpy array for realtime pipeline.
    d defines pipeline state. assumes segmenttimes defined by RT.set_pipeline.
    """

    sdmfile = d['workdir'] + d['filename']

    # define integration range
    if segment != -1:
        assert d.has_key('segmenttimes')
        d['segment'] = segment

        t0 = d['segmenttimes'][segment][0]
        t1 = d['segmenttimes'][segment][1]
        readints = n.round(24*3600*(t1 - t0)/d['inttime'], 0).astype(int)
        nskip = n.round(24*3600*(t0 - d['starttime_mjd'])/d['inttime'], 0).astype(int)
        print 'Reading segment %d/%d, times %s to %s' % (segment, len(d['segmenttimes'])-1, qa.time(qa.quantity(t0,'d'),form=['hms'], prec=9)[0], qa.time(qa.quantity(t1, 'd'), form=['hms'], prec=9)[0])
    else:
        nskip = 0
        readints = 0

    # read (all) data
    data = sdmreader.read_bdf(sdmfile, d['scan'], nskip=nskip, readints=readints).astype('complex64')

    # test that spw are in freq sorted order
    dfreq = n.array([d['spw_reffreq'][i+1] - d['spw_reffreq'][i] for i in range(len(d['spw_reffreq'])-1)])
    if not n.all(dfreq > 0):   # if spw not in freq order, then try to reorganize data
        print 'BDF spw frequencies out of order:', d['spw_reffreq']
        # use case 1: spw are rolled
        assert len(n.where(dfreq < 0)[0]) == 1
        print 'BDF spw frequency order rolled. Fixing...'
        rollch = n.sum([d['spw_nchan'][ss] for ss in range(n.where(dfreq < 0)[0][0]+1)])
        data = n.roll(data, rollch, axis=2)

    # optionally integrate (downsample)
    if d['read_downsample'] > 1:
        print 'Downsampling by factor of %d' % d['read_downsample']
        newsize = len(data)/d['read_downsample']
        sh = data.shape
        data2 = n.zeros( (newsize, sh[1], sh[2], sh[3]) )
        for i in range(newsize):
            data2[i] = data[i*d['read_downsample']:(i+1)*d['read_downsample']].mean(axis=0)
        data = data2

    return data.take(d['chans'], axis=2)

def get_uvw_segment(d, segment=-1):
    """ Calculates uvw for each baseline in a given segment.
    d defines pipeline state. assumes segmenttimes defined by RT.set_pipeline.
    """

    sdmfile = d['workdir'] + d['filename']

    # define times to read
    if segment != -1:
        assert d.has_key('segmenttimes')

        d['segment'] = segment
        t0 = d['segmenttimes'][segment][0]
        t1 = d['segmenttimes'][segment][1]
        datetime = qa.time(qa.quantity((t1+t0)/2,'d'),form=['ymdhms'], prec=9)[0]
        print 'Calculating uvw for segment %d' % (d['segment'])
    else:
        datetime = 0

    (u, v, w) = sdmreader.calc_uvw(sdmfile, d['scan'], datetime=datetime)

    # cast to units of lambda at first channel. -1 keeps consistent with ms reading convention
    u = u * d['freq_orig'][0] * (1e9/3e8) * (-1)     
    v = v * d['freq_orig'][0] * (1e9/3e8) * (-1)
    w = w * d['freq_orig'][0] * (1e9/3e8) * (-1)

    return u.astype('float32'), v.astype('float32'), w.astype('float32')

def sdm2ms(sdmfile, msfile, scan, inttime='0'):
    """ Converts sdm to ms format for a single scan.
    msfile defines the name template for the ms. Should end in .ms, but "s<scan>" will be put in.
    scan is string of (sdm counted) scan number.
    inttime is string to feed to split command. gives option of integrated data down in time.
    """

    # fill ms file
    msfile2 = msfile.rstrip('.ms') + '_s' + scan + '.ms'
    if os.path.exists(msfile2):
        print '%s already set.' % msfile2
    else:
        print 'No %s found. Creating anew.' % msfile2
        if inttime != '0':
            print 'Filtering by int time.'
            subprocess.call(['asdm2MS', '--ocm co --icm co --lazy --scans', scan, sdmfile, 'tmp_'+msfile2])
            cfg = tasklib.SplitConfig()  # configure split
            cfg.vis = 'tmp_'+msfile2
            cfg.out = msfile2
            cfg.timebin=inttime
            cfg.col = 'data'
            cfg.antenna='*&*'  # discard autos
            tasklib.split(cfg)  # run task
            # clean up
            shutil.rmtree('tmp_'+msfile2)
        else:
            subprocess.call(['asdm2MS', '--ocm co --icm co --lazy --scans', scan, sdmfile, msfile2])

    return msfile2

def filter_scans(sdmfile, namefilter='', intentfilter=''):
    """ Parses xml in sdmfile to get scan info for those containing 'namefilter' and 'intentfilter'
    mostly replaced by sdmreader.read_metadata.
    """

    goodscans = {}
    # find scans
    sdm = sdmpy.SDM(sdmfile)
    scans = [ (int(scan.scanNumber), str(scan.sourceName), str(scan.scanIntent)) for scan in sdm['Scan'] ]
    # set total number of integrations
    scanint = [int(scan.numIntegration) for scan in sdm['Main']]
    bdfnum = [int(scan.dataUID).split('/')[-1] for scan in sdm['Main']]
    for i in range(len(scans)):
        if intentfilter in scans[i][2]:
            if namefilter in scans[i][1]:
                goodscans[scans[i][0]] = (scans[i][1], scanint[i], bdfnum[i])
    print 'Found a total of %d scans and %d with name=%s and intent=%s.' % (len(scans), len(goodscans), namefilter, intentfilter)
    return goodscans

""" Misc stuff from Steve. Not yet in sdmreader
"""

def call_qatime(arg, form='', prec=0):
    """
    This is a wrapper for qa.time(), which in casa 4.0 returns a list of 
    strings instead of just a scalar string.  In this case, return the first 
    value in the list.
    - Todd Hunter
    """

    result = qa.time(arg, form=form, prec=prec)
    if (type(result) == list or type(result)==np.ndarray):
        return(result[0])
    else:
        return(result)

def listscans(dicts):
    myscans = dicts[0]
    mysources = dicts[1]
    if (myscans == []): return
    # Loop over scans
    for key in myscans.keys():
        mys = myscans[key]
        src = mys['source']
        tim = mys['timerange']
        sint= mys['intent']
        dur = mys['duration']*1440
        print '%8i %24s %48s  %.1f minutes  %s ' % (key, src, tim, dur, sint)
    durations = duration(myscans)
    print '  Found ', len(mysources),' sources in Source.xml'
    for key in durations:
        for mysrc in mysources.keys():
#            if (key[0] == mysources[mysrc]['sourceName']):
            if (key[0] == mysources[mysrc]['source']):
                ra = mysources[mysrc]['ra']
                dec = mysources[mysrc]['dec']
                directionCode = mysources[mysrc]['directionCode']
                break
        raString = qa.formxxx('%.12frad'%ra,format('hms'))
        decString = qa.formxxx('%.12frad'%dec,format('dms')).replace('.',':',2)
        print '   Total %24s (%d)  %5.1f minutes  (%.3f, %+.3f radian) %s: %s %s' % (key[0], int(mysrc), key[1], ra, dec, directionCode, raString, decString)
    durations = duration(myscans,nocal=True)
    for key in durations:
        print '   Total %24s      %5.1f minutes (neglecting pntg, atm & sideband cal. scans)' % (key[0],key[1])
    return
# Done

def duration(myscans, nocal=False):
    durations = []
    for key in myscans.keys():
        mys = myscans[key]
        src = mys['source']
        if (nocal and (mys['intent'].find('CALIBRATE_SIDEBAND')>=0 or
                       mys['intent'].find('CALIBRATE_POINTING')>=0 or
                       mys['intent'].find('CALIBRATE_ATMOSPHERE')>=0)):
            dur = 0
        else:
            dur = mys['duration']*1440
        new = 1
        for s in range(len(durations)):
            if (src == durations[s][0]):
                new = 0
                source = s
        if (new == 1):
            durations.append([src,dur])
        else:
            durations[source][1] = durations[source][1] + dur
    return(durations)
    
def readrx(sdmfile):

    # read Scan.xml into dictionary also and make a list
    xmlrx = minidom.parse(sdmfile+'/Receiver.xml')
    rxdict = {}
    rxlist = []
    rowlist = xmlrx.getElementsByTagName("row")
    for rownode in rowlist:
        a = rownode.getElementsByTagName("*")
        rowrxid = rownode.getElementsByTagName("receiverId")
        rxid = int(rowrxid[0].childNodes[0].nodeValue)
        rowfreqband = rownode.getElementsByTagName("frequencyBand")
        freqband = str(rowfreqband[0].childNodes[0].nodeValue)
        print "rxid = %d, freqband = %s" % (rxid,freqband)
    # return the dictionary for later use
    return rxdict
