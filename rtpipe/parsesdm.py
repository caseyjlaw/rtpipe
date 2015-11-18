#
# functions to read and convert sdm files
# claw, 14jun03
#

import logging
logger = logging.getLogger(__name__)
import numpy as n
import os, shutil, subprocess, glob, string
try:
    import casautil, tasklib
except ImportError:
    import pwkit.environments.casa.tasks as tasklib
    import pwkit.environments.casa.util as casautil
import sdmreader, sdmpy
import rtpipe.parseparams as pp
from rtlib_cython import calc_blarr
qa = casautil.tools.quanta()
me = casautil.tools.measures()

def get_metadata(filename, scan, paramfile='', **kwargs):
    """ Parses sdm file to define metadata for observation, including scan info, image grid parameters, pipeline memory usage, etc.
    Mirrors parsems.get_metadata().
    If paramfile defined, it will use it (filename or RT.Params instance ok).
    """

    # create primary state dictionary
    d = {}

    # set workdir
    d['filename'] = os.path.abspath(filename)
    d['workdir'] = os.path.dirname(d['filename'])

    # define parameters of pipeline via Params object
    params = pp.Params(paramfile)
    for k in params.defined:   # fill in default params
        d[k] = params[k]

    # overload with provided kwargs
    for key in kwargs.keys():
        logger.info('Setting %s to %s' % (key, kwargs[key]))
        d[key] = kwargs[key]

    if 'silent' in kwargs.keys():
        loglevel = logging.ERROR
    else:
        loglevel = logging.INFO
    logger.setLevel(loglevel)

    # define scan list
    if not d.has_key('bdfdir'): d['bdfdir'] = None
    scans, sources = sdmreader.read_metadata(d['filename'], scan, bdfdir=d['bdfdir'])

    # define source props
    d['source'] = scans[scan]['source']
    d['radec'] = [(prop['ra'], prop['dec']) for (sr,prop) in sources.iteritems() if prop['source'] == d['source']][0]

    # define spectral info
    sdm = sdmpy.SDM(d['filename'])
    d['spw_orig'] = [int(row.spectralWindowId.split('_')[1]) for row in sdm['SpectralWindow']]
    d['spw_nchan'] = [int(row.numChan) for row in sdm['SpectralWindow']]

    try:
        d['spw_reffreq'] = [float(row.chanFreqStart) for row in sdm['SpectralWindow']]   # nominal
    except:
        d['spw_reffreq'] = [float(row.chanFreqArray.strip().split(' ')[2]) for row in sdm['SpectralWindow']]  # GMRT uses array of all channel starts
    try:
        d['spw_chansize'] = [float(row.chanFreqStep) for row in sdm['SpectralWindow']]   # nominal
    except:
        d['spw_chansize'] = [float(row.chanWidthArray.strip().split(' ')[2]) for row in sdm['SpectralWindow']]   # GMRT uses array of all channel starts

    # select spw. note that spw selection not fully supported yet.
    if not len(d['spw']):
        d['spw'] = d['spw_orig']

    spwch = []
    reffreq = d['spw_reffreq']; spectralwindow = d['spw_orig']; numchan = d['spw_nchan']; chansize = d['spw_chansize']
    for freq in sorted(d['spw_reffreq']):
        ii = reffreq.index(freq)
        if spectralwindow[ii] in d['spw']:
            spwch.extend(list(n.linspace(reffreq[ii], reffreq[ii]+(numchan[ii]-1)*chansize[ii], numchan[ii])))  # spacing of channel *centers*

    d['freq_orig'] = n.array([n.mean(spwch[i:i+d['read_fdownsample']]) for i in range(0, len(spwch), d['read_fdownsample'])], dtype='float32')/1e9
#    d['freq_orig'] = n.array(spwch, dtype='float32')/1e9  # without downsample

    # select subset of channels
    if not len(d['chans']):
        d['chans'] = range(len(d['freq_orig']))

    d['nspw'] = len(d['spw'])
    d['freq'] = d['freq_orig'][d['chans']]
    d['nchan'] = len(d['chans'])

    # define chan ranges per spw (before selecting subset)
    spw_chanr = []; i0=0
    for nch in d['spw_nchan']:
        spw_chanr.append((i0, i0+nch))
        i0 = nch
    d['spw_chanr'] = spw_chanr

    # define nchan per spw after selecting subset
    d['spw_nchan_select'] = [len([ch for ch in range(d['spw_chanr'][i][0], d['spw_chanr'][i][1]) if ch in d['chans']]) for i in range(len(d['spw_chanr']))]
    spw_chanr_select = []; i0=0
    for nch in d['spw_nchan_select']:
        spw_chanr_select.append((i0, i0+nch))
        i0 = nch
    d['spw_chanr_select'] = spw_chanr_select

    # define image params
    d['urange'] = {}; d['vrange'] = {}
    d['scan'] = scan
    (u, v, w) = sdmreader.calc_uvw(d['filename'], d['scan'])  # default uses time at start
    u = u * d['freq_orig'][0] * (1e9/3e8) * (-1)     
    v = v * d['freq_orig'][0] * (1e9/3e8) * (-1)     
    d['urange'][d['scan']] = u.max() - u.min()
    d['vrange'][d['scan']] = v.max() - v.min()
    d['dishdiameter'] = float(sdm['Antenna'][0].dishDiameter.strip())  # should be in meters
    d['uvres_full'] = n.round(d['dishdiameter']/(3e-1/d['freq'].min())/2).astype('int')    # delay beam larger than VLA field of view at all freqs. assumes freq in GHz.
    # **this may let vis slip out of bounds. should really define grid out to 2*max(abs(u)) and 2*max(abs(v)). in practice, very few are lost.**

    if not all([d.has_key('npixx_full'), d.has_key('npixy_full')]):
        urange = d['urange'][d['scan']]*(d['freq'].max()/d['freq_orig'][0])   # uvw from get_uvw already in lambda at ch0
        vrange = d['vrange'][d['scan']]*(d['freq'].max()/d['freq_orig'][0])
        powers = n.fromfunction(lambda i,j: 2**i*3**j, (14,10), dtype='int')   # power array for 2**i * 3**j
        rangex = n.round(d['uvoversample']*urange).astype('int')
        rangey = n.round(d['uvoversample']*vrange).astype('int')
        largerx = n.where(powers-rangex/d['uvres_full'] > 0, powers, powers[-1,-1])
        p2x, p3x = n.where(largerx == largerx.min())
        largery = n.where(powers-rangey/d['uvres_full'] > 0, powers, powers[-1,-1])
        p2y, p3y = n.where(largery == largery.min())
        d['npixx_full'] = (2**p2x * 3**p3x)[0]
        d['npixy_full'] = (2**p2y * 3**p3y)[0]

    # define ants/bls
    # hacking here to fit observatory-specific use of antenna names
    if 'VLA' in sdm['ExecBlock'][0]['telescopeName']:
        d['ants'] = [int(ant.name.lstrip('ea')) for ant in sdm['Antenna']]    # Not complete. Execblock defines ants per scan, which can change.
    elif 'GMRT' in sdm['ExecBlock'][0]['telescopeName']:        
        d['ants'] = [int(ant.antennaId.split('_')[1]) for ant in sdm['Antenna']]

   # remove unwanted ants
    for ant in d['excludeants']:
        d['ants'].remove(ant)
    d['ants'].sort()  # required to assure that added antennas don't confuse cal antenna parsing

    d['nants'] = len(d['ants'])
#    d['blarr'] = n.array([[d['ants'][i],d['ants'][j]] for j in range(d['nants']) for i in range(0,j)])
    d['nbl'] = d['nants']*(d['nants']-1)/2

    # define times
    d['starttime_mjd'] = scans[d['scan']]['startmjd']
    # assume inttime same for all scans

    for scan in sdm['Main']:
        interval = float(scan.interval)
        nints = int(scan.numIntegration)
        # get inttime in seconds
        if 'VLA' in sdm['ExecBlock'][0]['telescopeName']:
            inttime = interval/nints          # VLA uses interval as scan duration
            scannum = int(scan.scanNumber)
        elif 'GMRT' in sdm['ExecBlock'][0]['telescopeName']:        
            inttime = interval                # GMRT uses interval as the integration duration
            scannum = int(scan.subscanNumber)
        if scannum == d['scan']:
            d['inttime'] = n.round(inttime)*1e-9
            d['nints'] = nints

    # define pols
    d['pols_orig'] = [pol for pol in sdm['Polarization'][0].corrType.strip().split(' ') if pol in ['XX', 'YY', 'XY', 'YX', 'RR', 'LL', 'RL', 'LR']]
    d['npol_orig'] = int(sdm['Polarization'][0].numCorr)

    # summarize metadata
    logger.info('\n')
    logger.info('Metadata summary:')
    logger.info('\t Working directory and data at %s, %s' % (d['workdir'], os.path.basename(d['filename'])))
    logger.info('\t Using scan %d, source %s' % (int(d['scan']), d['source']))
    logger.info('\t nants, nbl: %d, %d' % (d['nants'], d['nbl']))
    logger.info('\t Freq range (%.3f -- %.3f). %d spw with %d chans.' % (d['freq'].min(), d['freq'].max(), d['nspw'], d['nchan']))
    logger.info('\t Scan has %d ints (%.1f s) and inttime %.3f s' % (d['nints'], d['nints']*d['inttime'], d['inttime']))
    logger.info('\t %d polarizations: %s' % (d['npol_orig'], d['pols_orig']))
    logger.info('\t Ideal uvgrid npix=(%d,%d) and res=%d (oversample %.1f)' % (d['npixx_full'], d['npixy_full'], d['uvres_full'], d['uvoversample']))

    return d

def read_bdf_segment(d, segment=-1):
    """ Reads bdf (sdm) format data into numpy array for realtime pipeline.
    d defines pipeline state. assumes segmenttimes defined by RT.set_pipeline.
    d should have 'writebdfpkl' key to define boolean for writing to bdfpkls in ASDMBinary directory.
    """

    # define integration range
    if segment != -1:
        assert d.has_key('segmenttimes'), 'd must have segmenttimes defined'
        assert segment < d['nsegments'], 'segment %d is too big for nsegments %d' % (segment, d['nsegments'])
        readints = d['readints']
        nskip = (24*3600*(d['segmenttimes'][segment,0] - d['starttime_mjd'])/d['inttime']).astype(int)
        logger.info('Reading segment %d/%d, times %s to %s' % (segment, len(d['segmenttimes'])-1, qa.time(qa.quantity(d['segmenttimes'][segment,0],'d'),form=['hms'], prec=9)[0], qa.time(qa.quantity(d['segmenttimes'][segment,1], 'd'), form=['hms'], prec=9)[0]))
    else:
        nskip = 0
        readints = 0

    # read (all) data
    if not d.has_key('bdfdir'): d['bdfdir'] = None
    data = sdmreader.read_bdf(d['filename'], d['scan'], nskip=nskip, readints=readints, writebdfpkl=d['writebdfpkl'], bdfdir=d['bdfdir']).astype('complex64')

    # read Flag.xml and apply flags for given ant/time range
    if d['applyonlineflags'] and segment > -1:   # currently only implemented for segmented data
        sdm = sdmpy.SDM(d['filename'])
        antdict = dict(zip([ant.antennaId for ant in sdm['Antenna']], [int(ant.name.lstrip('ea')) for ant in sdm['Antenna']]))
        antflags = [(antdict[flag.antennaId.split(' ')[2]], int(flag.startTime)/(1e9*24*3600), int(flag.endTime)/(1e9*24*3600)) for flag in sdm['Flag']]  # assumes one flag per entry
        logger.info('Found online flags for %d antenna/time ranges.' % (len(antflags)))
        blarr = calc_blarr(d)
        timearr = n.linspace(d['segmenttimes'][segment][0], d['segmenttimes'][segment][1], d['readints'])
        badints_cum = []
        for antflag in antflags:
            antnum, time0, time1 = antflag
            badbls = n.where((blarr == antnum).any(axis=1))[0]
            badints = n.where( (timearr >= time0) & (timearr <= time1) )[0]
            logger.debug('Flagging %d ints for antnum %d' % (len(badints), antnum))
            for badint in badints:
                data[badint, badbls] = 0j
            badints_cum = badints_cum + list(badints)
        logger.info('Applied online flags to %d ints.' % (len(set(badints_cum))))
    elif segment == -1:
        logger.info('Online flags not yet supported without segments.')
    else:
        logger.info('Not applying online flags.')

    # test that spw are in freq sorted order
    # only one use case supported: rolled spw
    dfreq = n.array([d['spw_reffreq'][i+1] - d['spw_reffreq'][i] for i in range(len(d['spw_reffreq'])-1)])
    dfreqneg = [df for df in dfreq if df < 0]
    if len(dfreqneg) <= 1:      # if spw are permuted, then roll them. !! not a perfect test of permutability!!
        if len(dfreqneg) == 1:
            logger.warn('Rolling spw frequencies to increasing order: %s' % str(d['spw_reffreq']))
            rollch = n.sum([d['spw_nchan'][ss] for ss in range(n.where(dfreq < 0)[0][0]+1)])
            data = n.roll(data, rollch, axis=2)
    else:
        raise StandardError, 'SPW out of order and can\'t be permuted to increasing order: %s' % str(d['spw_reffreq'])

    # optionally integrate (downsample)
    if ((d['read_tdownsample'] > 1) or (d['read_fdownsample'] > 1)):
        sh = data.shape
        tsize = sh[0]/d['read_tdownsample']
        fsize = sh[2]/d['read_fdownsample']
        data2 = n.zeros( (tsize, sh[1], fsize, sh[3]), dtype='complex64')
        if d['read_tdownsample'] > 1:
            logger.info('Downsampling in time by %d' % d['read_tdownsample'])
            for i in range(tsize):
                data2[i] = data[i*d['read_tdownsample']:(i+1)*d['read_tdownsample']].mean(axis=0)
        if d['read_fdownsample'] > 1:
            logger.info('Downsampling in frequency by %d' % d['read_fdownsample'])
            for i in range(fsize):
                data2[:,:,i,:] = data[:,:,i*d['read_fdownsample']:(i+1)*d['read_fdownsample']].mean(axis=2)
        data = data2

    takepol = [d['pols_orig'].index(pol) for pol in d['pols']]
    logger.debug('Selecting pols %s' % d['pols'])

    return data.take(d['chans'], axis=2).take(takepol, axis=3)

def get_uvw_segment(d, segment=-1):
    """ Calculates uvw for each baseline at mid time of a given segment.
    d defines pipeline state. assumes segmenttimes defined by RT.set_pipeline.
    """

    # define times to read
    if segment != -1:
        assert d.has_key('segmenttimes'), 'd must have segmenttimes defined'

        t0 = d['segmenttimes'][segment][0]
        t1 = d['segmenttimes'][segment][1]
        datetime = qa.time(qa.quantity((t1+t0)/2,'d'),form=['ymdhms'], prec=9)[0]
        logger.info('Calculating uvw for segment %d' % (segment))
    else:
        datetime = 0

    (u, v, w) = sdmreader.calc_uvw(d['filename'], d['scan'], datetime=datetime)

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

    sdmdir = os.path.dirname(sdmfile)
    msdir = os.path.dirname(msfile)

    # fill ms file
#    msfile2 = msfile.rstrip('.ms') + '_s' + scan + '.ms'
    if os.path.exists(msfile):
        logger.debug('%s already set.' % msfile)
    else:
        logger.info('No %s found. Creating anew.' % msfile)
        if inttime != '0':
            logger.info('Filtering by int time.')
            subprocess.call(['asdm2MS', '--ocm', 'co', '--icm', 'co', '--lazy', '--scans', scan, sdmfile, msfile + '.tmp'])
            cfg = tasklib.SplitConfig()  # configure split
            cfg.vis = msfile + '.tmp'
            cfg.out = msfile
            cfg.timebin=inttime
            cfg.col = 'data'
            cfg.antenna='*&*'  # discard autos
            tasklib.split(cfg)  # run task
            # clean up
            shutil.rmtree(msfile+'.tmp')
        else:
            subprocess.call(['asdm2MS', '--ocm', 'co', '--icm', 'co', '--lazy', '--scans', scan, sdmfile, msfile])

    return msfile

def filter_scans(sdmfile, namefilter='', intentfilter=''):
    """ Parses xml in sdmfile to get scan info for those containing 'namefilter' and 'intentfilter'
    mostly replaced by sdmreader.read_metadata.
    """

    goodscans = {}
    # find scans
    sdm = sdmpy.SDM(sdmfile)

    if 'VLA' in sdm['ExecBlock'][0]['telescopeName']:
        scans = [ (int(scan.scanNumber), str(scan.sourceName), str(scan.scanIntent)) for scan in sdm['Scan'] ]
    elif 'GMRT' in sdm['ExecBlock'][0]['telescopeName']:        
        scans = [ (int(scan.scanNumber), str(scan.sourceName), str(scan.scanIntent)) for scan in sdm['Subscan'] ]

    # set total number of integrations
    scanint = [int(scan.numIntegration) for scan in sdm['Main']]
    bdfnum = [scan.dataUID.split('/')[-1] for scan in sdm['Main']]
    for i in range(len(scans)):
        if intentfilter in scans[i][2]:
            if namefilter in scans[i][1]:
                goodscans[scans[i][0]] = (scans[i][1], scanint[i], bdfnum[i])
    logger.debug('Found a total of %d scans and %d with name=%s and intent=%s.' % (len(scans), len(goodscans), namefilter, intentfilter))
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
        logger.debug('%8i %24s %48s  %.1f minutes  %s ' % (key, src, tim, dur, sint))
    durations = duration(myscans)
    logger.debug('  Found ', len(mysources),' sources in Source.xml')
    for key in durations:
        for mysrc in mysources.keys():
#            if (key[0] == mysources[mysrc]['sourceName']):
            if (key[0] == mysources[mysrc]['source']):
                ra = mysources[mysrc]['ra']
                dec = mysources[mysrc]['dec']
                break
        raString = qa.formxxx('%.12frad'%ra,format('hms'))
        decString = qa.formxxx('%.12frad'%dec,format('dms')).replace('.',':',2)
        logger.debug('   Total %24s (%d)  %5.1f minutes  (%.3f, %+.3f radian): %s %s' % (key[0], int(mysrc), key[1], ra, dec, raString, decString))
    durations = duration(myscans,nocal=True)
    for key in durations:
        logger.debug('   Total %24s      %5.1f minutes (neglecting pntg, atm & sideband cal. scans)' % (key[0],key[1]))
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
        logger.debug("rxid = %d, freqband = %s" % (rxid,freqband))
    # return the dictionary for later use
    return rxdict
