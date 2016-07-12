import logging
logger = logging.getLogger(__name__)
import numpy as np
import os.path
from shutil import rmtree
from subprocess import call
from time import time
import pwkit.environments.casa.tasks as tasklib
import pwkit.environments.casa.util as casautil
import sdmpy
import rtpipe.parseparams as pp
from rtlib_cython import calc_blarr
from lxml.etree import XMLSyntaxError
qa = casautil.tools.quanta()
me = casautil.tools.measures()


def get_metadata(filename, scan, paramfile='', **kwargs):
    """ Parses sdm file to define metadata for observation, including scan info,
    image grid parameters, pipeline memory usage, etc.
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
        if key in params.defined:
            stdname = '(standard)'
        else:
            stdname = ''
        logger.info('Setting %s key %s to %s' % (stdname, key, kwargs[key]))
        d[key] = kwargs[key]

    # option of not writing log file (need to improve later)
    if d['logfile']:
        fh = logging.FileHandler(os.path.join(d['workdir'], 'rtpipe_%d.log' % int(round(time()))))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.parent.addHandler(fh)
        if hasattr(logging, d['loglevel']):
            logger.parent.setLevel(getattr(logging, d['loglevel']))
        else:
            logger.warn('loglevel of {0} not attribute of logging'.format(d['loglevel']))

    # define scan list
    if 'bdfdir' not in d:
        d['bdfdir'] = ''

    scans = read_scans(d['filename'], bdfdir=d['bdfdir'])
    sources = read_sources(d['filename'])

    # define source props
    d['source'] = scans[scan]['source']
    d['radec'] = [(prop['ra'], prop['dec'])
                  for (sr, prop) in sources.iteritems()
                  if prop['source'] == d['source']][0]

    # define spectral info
    sdm = getsdm(d['filename'])
    d['spw_orig'] = [int(str(row.spectralWindowId).split('_')[1])
                     for row in sdm['SpectralWindow']]
    d['spw_nchan'] = [int(row.numChan) for row in sdm['SpectralWindow']]

    try:
        d['spw_reffreq'] = [float(row.chanFreqStart)
                            for row in sdm['SpectralWindow']]   # nominal
    except:
        # GMRT uses array of all channel starts
        d['spw_reffreq'] = [float(row.chanFreqArray.strip().split(' ')[2])
                            for row in sdm['SpectralWindow']]
    try:
        d['spw_chansize'] = [float(row.chanFreqStep)
                             for row in sdm['SpectralWindow']]   # nominal
    except:
        # GMRT uses array of all channel starts
        d['spw_chansize'] = [float(row.chanWidthArray.strip().split(' ')[2])
                             for row in sdm['SpectralWindow']]

    # select spw. note that spw selection not fully supported yet.
    if not len(d['spw']):
        d['spw'] = d['spw_orig']

    spwch = []
    reffreq = d['spw_reffreq']
    spectralwindow = d['spw_orig']
    numchan = d['spw_nchan']
    chansize = d['spw_chansize']
    for freq in sorted(d['spw_reffreq']):
        ii = reffreq.index(freq)
        if spectralwindow[ii] in d['spw']:
            # spacing of channel *centers*
            spwch.extend(list(np.linspace(reffreq[ii], reffreq[ii]
                                          + (numchan[ii]-1) * chansize[ii],
                                          numchan[ii])))

    d['freq_orig'] = np.array([np.mean(spwch[i:i+d['read_fdownsample']])
                               for i in range(0, len(spwch),
                                              d['read_fdownsample'])],
                              dtype='float32') / 1e9

    # select subset of channels
    if not len(d['chans']):
        d['chans'] = range(len(d['freq_orig']))

    d['nspw'] = len(d['spw'])
    d['freq'] = d['freq_orig'][d['chans']]
    d['nchan'] = len(d['chans'])

    # define chan ranges per spw (before selecting subset)
    spw_chanr = []
    i0 = 0
    for nch in d['spw_nchan']:
        spw_chanr.append((i0, i0+nch))
        i0 = nch
    d['spw_chanr'] = spw_chanr

    # define nchan per spw after selecting subset
    d['spw_nchan_select'] = [len([ch
                                  for ch in range(d['spw_chanr'][i][0],
                                                  d['spw_chanr'][i][1])
                                  if ch in d['chans']])
                             for i in range(len(d['spw_chanr']))]
    spw_chanr_select = []
    i0 = 0
    for nch in d['spw_nchan_select']:
        spw_chanr_select.append((i0, i0+nch))
        i0 += nch
    d['spw_chanr_select'] = spw_chanr_select

    # define image params
    d['urange'] = {}
    d['vrange'] = {}
    d['scan'] = scan
    (u, v, w) = calc_uvw(d['filename'],
                         d['scan'], bdfdir=d['bdfdir'])  # default uses time at start
    u = u * d['freq_orig'][0] * (1e9/3e8) * (-1)
    v = v * d['freq_orig'][0] * (1e9/3e8) * (-1)
    d['urange'][d['scan']] = u.max() - u.min()
    d['vrange'][d['scan']] = v.max() - v.min()
    d['dishdiameter'] = float(str(sdm['Antenna'][0]
                                  .dishDiameter).strip())  # should be in meters
    # delay beam larger than VLA field of view at all freqs
    d['uvres_full'] = np.round(d['dishdiameter']
                               / (3e-1 / d['freq'].min()) / 2).astype('int')

    if not all(('npixx_full' in d, 'npixy_full' in d)):
        # uvw from get_uvw already in lambda at ch0
        urange = d['urange'][d['scan']] * (d['freq'].max()
                                           / d['freq_orig'][0])
        vrange = d['vrange'][d['scan']] * (d['freq'].max()
                                           / d['freq_orig'][0])
        # power array for 2**i * 3**j
        powers = np.fromfunction(lambda i, j: 2**i*3**j,
                                 (14, 10), dtype='int')
        rangex = np.round(d['uvoversample']*urange).astype('int')
        rangey = np.round(d['uvoversample']*vrange).astype('int')
        largerx = np.where(powers - rangex / d['uvres_full'] > 0,
                           powers, powers[-1, -1])
        p2x, p3x = np.where(largerx == largerx.min())
        largery = np.where(powers - rangey / d['uvres_full'] > 0,
                           powers, powers[-1, -1])
        p2y, p3y = np.where(largery == largery.min())
        d['npixx_full'] = (2**p2x * 3**p3x)[0]
        d['npixy_full'] = (2**p2y * 3**p3y)[0]

    # define ants/bls
    # hacking here to fit observatory-specific use of antenna names
    if 'VLA' in str(sdm['ExecBlock'][0]['telescopeName']):
# find config first, then antids, then ant names
        configid = [str(row.configDescriptionId) for row in sdm['Main']
                    if d['scan'] == int(row.scanNumber)][0]
        antids = [str(row.antennaId) for row in sdm['ConfigDescription']
                  if configid == row.configDescriptionId][0].split(' ')[2:]
        d['ants'] = [int(str(row.name).lstrip('ea'))
                     for antid in antids
                     for row in sdm['Antenna']
                     if antid == str(row.antennaId)]
# Not complete. Execblock defines ants per scan, which can change.
#        d['ants'] = [int(ant.name.lstrip('ea'))
#                     for ant in sdm['Antenna']]
    elif 'GMRT' in str(sdm['ExecBlock'][0]['telescopeName']):
        d['ants'] = [int(str(ant.antennaId).split('_')[1])
                     for ant in sdm['Antenna']]

    # remove unwanted ants
    for ant in d['excludeants']:
        d['ants'].remove(ant)
    # required to assure that added antennas don't confuse cal antenna parsing
    d['ants'].sort()

    d['nants'] = len(d['ants'])
#    d['blarr'] = np.array([[d['ants'][i],d['ants'][j]]
#                           for j in range(d['nants']) for i in range(0,j)])
    d['nbl'] = d['nants']*(d['nants']-1)/2

    # define times
    d['starttime_mjd'] = scans[d['scan']]['startmjd']

    # assume inttime same for all scans
    scan = sdm.scan(d['scan'])
    d['inttime'] = scan.bdf.get_integration(0).interval
    d['nints'] = int(scan.bdf.numIntegration)

    # define pols
    d['pols_orig'] = [pol
                      for pol in (str(sdm['Polarization'][0]
                                      .corrType).strip()
                                      .split(' '))
                      if pol in ['XX', 'YY', 'XY', 'YX',
                                 'RR', 'LL', 'RL', 'LR']]
    d['npol_orig'] = int(sdm['Polarization'][0].numCorr)

    # summarize metadata
    logger.info('\n')
    logger.info('Metadata summary:')
    logger.info('\t Working directory and data at %s, %s'
                % (d['workdir'], os.path.basename(d['filename'])))
    logger.info('\t Using scan %d, source %s'
                % (int(d['scan']), d['source']))
    logger.info('\t nants, nbl: %d, %d' % (d['nants'], d['nbl']))
    logger.info('\t Freq range (%.3f -- %.3f). %d spw with %d chans.'
                % (d['freq'].min(), d['freq'].max(), d['nspw'], d['nchan']))
    logger.info('\t Scan has %d ints (%.1f s) and inttime %.3f s'
                % (d['nints'], d['nints']*d['inttime'], d['inttime']))
    logger.info('\t %d polarizations: %s'
                % (d['npol_orig'], d['pols_orig']))
    logger.info('\t Ideal uvgrid npix=(%d,%d) and res=%d (oversample %.1f)'
                % (d['npixx_full'], d['npixy_full'], d['uvres_full'],
                   d['uvoversample']))

    return d


def read_bdf(sdmfile, scannum, nskip=0, readints=0, bdfdir=''):
    """ Uses sdmpy to read a given range of integrations from sdm of given scan.

    readints=0 will read all of bdf (skipping nskip).
    """

    assert os.path.exists(sdmfile), 'sdmfile %s does not exist' % sdmfile

    sdm = getsdm(sdmfile, bdfdir=bdfdir)
    scan = sdm.scan(scannum)
    assert scan.bdf.fname, 'bdfstr not defined for scan %d' % scannum

    if readints == 0:
        readints = scan.bdf.numIntegration - nskip

    logger.info('Reading %d ints starting at int %d' % (readints, nskip))
    npols = len(sdmpy.scan.sdmarray(sdm['Polarization'][0].corrType))
    data = np.empty( (readints, scan.bdf.numBaseline, sum(scan.numchans), npols), dtype='complex64', order='C')
    data[:] = scan.bdf.get_data(trange=[nskip, nskip+readints]).reshape(data.shape)

    return data


def read_bdf_segment(d, segment=-1):
    """ Reads bdf (sdm) format data into numpy array for realtime pipeline.

    d defines pipeline state. assumes segmenttimes defined by RT.set_pipeline.
    d should have 'writebdfpkl' key to define
    boolean for writing to bdfpkls in ASDMBinary directory.
    """

    # define integration range
    if segment != -1:
        assert 'segmenttimes' in d
        assert segment < d['nsegments'], \
            'segment %d is too big for nsegments %d' % (segment,
                                                        d['nsegments'])
        readints = d['readints']
        nskip = (24*3600*(d['segmenttimes'][segment, 0]
                          - d['starttime_mjd'])
                 / d['inttime']).astype(int)
        logger.info('Reading scan %d, segment %d/%d, times %s to %s'
                    % (d['scan'], segment,
                       len(d['segmenttimes'])-1,
                       qa.time(qa.quantity(d['segmenttimes'][segment, 0], 'd'),
                               form=['hms'], prec=9)[0],
                       qa.time(qa.quantity(d['segmenttimes'][segment, 1], 'd'),
                               form=['hms'], prec=9)[0]))
    else:
        nskip = 0
        readints = 0

    # read (all) data
    if 'bdfdir' not in d:
        d['bdfdir'] = None

    data = read_bdf(d['filename'], d['scan'], nskip=nskip, readints=readints, bdfdir=d['bdfdir']).astype('complex64')

    # read Flag.xml and apply flags for given ant/time range
    # currently only implemented for segmented data
    if d['applyonlineflags'] and segment > -1:
        sdm = getsdm(d['filename'])

        allantdict = dict(zip([str(ant.antennaId) for ant in sdm['Antenna']],
                              [int(str(ant.name).lstrip('ea'))
                               for ant in sdm['Antenna']]))
        antflags = [(allantdict[str(flag.antennaId).split(' ')[2]],
                     int(flag.startTime)/(1e9*24*3600),
                     int(flag.endTime)/(1e9*24*3600))
                    for flag in sdm['Flag']]  # assumes one flag per entry
        logger.info('Found online flags for %d antenna/time ranges.'
                    % (len(antflags)))
        blarr = calc_blarr(d)  # d may define different ants than in allantdict
        timearr = np.linspace(d['segmenttimes'][segment][0],
                              d['segmenttimes'][segment][1], d['readints'])
        badints_cum = []
        for antflag in antflags:
            antnum, time0, time1 = antflag
            badbls = np.where((blarr == antnum).any(axis=1))[0]
            badints = np.where((timearr >= time0) & (timearr <= time1))[0]
            for badint in badints:
                data[badint, badbls] = 0j
            badints_cum = badints_cum + list(badints)
        logger.info('Applied online flags to %d ints.'
                    % (len(set(badints_cum))))
    elif segment == -1:
        logger.warn('Online flags not yet supported without segments.')
    else:
        logger.info('Not applying online flags.')

    # test that spw are in freq sorted order
    # only one use case supported: rolled spw
    dfreq = np.array([d['spw_reffreq'][i+1] - d['spw_reffreq'][i]
                      for i in range(len(d['spw_reffreq'])-1)])
    dfreqneg = [df for df in dfreq if df < 0]
    # if spw are permuted, then roll them.
    # !! not a perfect test of permutability!!
    if len(dfreqneg) <= 1:
        if len(dfreqneg) == 1:
            logger.info('Rolling spw frequencies to increasing order: %s'
                        % str(d['spw_reffreq']))
            rollch = np.sum([d['spw_nchan'][ss]
                             for ss in range(np.where(dfreq < 0)[0][0]+1)])
            data = np.roll(data, rollch, axis=2)
    else:
        raise StandardError('SPW out of order and can\'t be permuted '
                            'to increasing order: %s'
                            % str(d['spw_reffreq']))

    # optionally integrate (downsample)
    if ((d['read_tdownsample'] > 1) or (d['read_fdownsample'] > 1)):
        sh = data.shape
        tsize = sh[0]/d['read_tdownsample']
        fsize = sh[2]/d['read_fdownsample']
        data2 = np.zeros((tsize, sh[1], fsize, sh[3]), dtype='complex64')
        if d['read_tdownsample'] > 1:
            logger.info('Downsampling in time by %d' % d['read_tdownsample'])
            for i in range(tsize):
                data2[i] = data[
                    i*d['read_tdownsample']:(i+1)*d['read_tdownsample']
                    ].mean(axis=0)
        if d['read_fdownsample'] > 1:
            logger.info('Downsampling in frequency by %d'
                        % d['read_fdownsample'])
            for i in range(fsize):
                data2[:, :, i, :] = data[
                    :, :, i * d['read_fdownsample']:(i+1)*d['read_fdownsample']
                    ].mean(axis=2)
        data = data2

    takepol = [d['pols_orig'].index(pol) for pol in d['pols']]
    logger.debug('Selecting pols %s' % d['pols'])

    return data.take(d['chans'], axis=2).take(takepol, axis=3)


def calc_uvw(sdmfile, scan=0, datetime=0, radec=(), bdfdir=''):
    """ Calculates and returns uvw in meters for a given SDM, time, and pointing direction.
    sdmfile is path to sdm directory that includes "Station.xml" file.
    scan is scan number defined by observatory.
    datetime is time (as string) to calculate uvw (format: '2014/09/03/08:33:04.20')
    radec is (ra,dec) as tuple in units of degrees (format: (180., +45.))
    bdfdir is path to bdfs (optional, for pre-archive SDMs)
    """

    assert os.path.exists(os.path.join(sdmfile, 'Station.xml')), 'sdmfile %s has no Station.xml file. Not an SDM?' % sdmfile

    # get scan info
    scans = read_scans(sdmfile, bdfdir=bdfdir)
    sources = read_sources(sdmfile)
#    scans, sources = read_metadata(sdmfile, scan)

    # default is to use scan info
    if (datetime == 0) and (len(radec) == 0):
        assert scan != 0, 'scan must be set when using datetime and radec'   # default scan value not valid

        logger.info('Calculating uvw for first integration of scan %d of source %s' % (scan, scans[scan]['source']))
        datetime = qa.time(qa.quantity(scans[scan]['startmjd'],'d'), form="ymd", prec=8)[0]
        sourcenum = [kk for kk in sources.keys() if sources[kk]['source'] == scans[scan]['source']][0]
        direction = me.direction('J2000', str(np.degrees(sources[sourcenum]['ra']))+'deg', str(np.degrees(sources[sourcenum]['dec']))+'deg')

    # secondary case is when datetime is also given
    elif (datetime != 0) and (len(radec) == 0):
        assert scan != 0, 'scan must be set when using datetime and radec'   # default scan value not valid
        assert '/' in datetime, 'datetime must be in yyyy/mm/dd/hh:mm:ss.sss format'

        logger.info('Calculating uvw at %s for scan %d of source %s' % (datetime, scan, scans[scan]['source']))
        sourcenum = [kk for kk in sources.keys() if sources[kk]['source'] == scans[scan]['source']][0]
        direction = me.direction('J2000', str(np.degrees(sources[sourcenum]['ra']))+'deg', str(np.degrees(sources[sourcenum]['dec']))+'deg')

    else:
        assert '/' in datetime, 'datetime must be in yyyy/mm/dd/hh:mm:ss.sss format'
        assert len(radec) == 2, 'radec must be (ra,dec) tuple in units of degrees'

        logger.info('Calculating uvw at %s in direction %s' % (datetime, direction))
        logger.info('This mode assumes all antennas used.')
        ra = radec[0]; dec = radec[1]
        direction = me.direction('J2000', str(ra)+'deg', str(dec)+'deg')

    # define metadata "frame" for uvw calculation
    sdm = getsdm(sdmfile)
    telescopename = str(sdm['ExecBlock'][0]['telescopeName']).strip()
    logger.debug('Found observatory name %s' % telescopename)

    me.doframe(me.observatory(telescopename))
    me.doframe(me.epoch('utc', datetime))
    me.doframe(direction)

    # read antpos
    if scan != 0:
        configid = [str(row.configDescriptionId) for row in sdm['Main'] if scan == int(row.scanNumber)][0]
        antidlist = [str(row.antennaId) for row in sdm['ConfigDescription'] if configid == str(row.configDescriptionId)][0].split(' ')[2:]
        stationidlist = [ant.stationId for antid in antidlist for ant in sdm['Antenna'] if antid == str(ant.antennaId)]
    else:
        stationidlist = [str(ant.stationId) for ant in sdm['Antenna']]

    positions = [str(station.position).strip().split(' ')
                 for station in sdm['Station'] 
                 if station.stationId in stationidlist]
    x = [float(positions[i][2]) for i in range(len(positions))]
    y = [float(positions[i][3]) for i in range(len(positions))]
    z = [float(positions[i][4]) for i in range(len(positions))]
    ants = me.position('itrf', qa.quantity(x, 'm'), qa.quantity(y, 'm'), qa.quantity(z, 'm'))

    # calc bl
    bls = me.asbaseline(ants)
    uvwlist = me.expand(me.touvw(bls)[0])[1]['value']

    # define new bl order to match sdm binary file bl order
    u = np.empty(len(uvwlist)/3); v = np.empty(len(uvwlist)/3); w = np.empty(len(uvwlist)/3)
    nants = len(ants['m0']['value'])
    ord1 = [i*nants+j for i in range(nants) for j in range(i+1,nants)]
    ord2 = [i*nants+j for j in range(nants) for i in range(j)]
    key=[]
    for new in ord2:
        key.append(ord1.index(new))
    for i in range(len(key)):
        u[i] = uvwlist[3*key[i]]
        v[i] = uvwlist[3*key[i]+1]
        w[i] = uvwlist[3*key[i]+2]

    return u, v, w


def get_uvw_segment(d, segment=-1):
    """ Calculates uvw for each baseline at mid time of a given segment.
    d defines pipeline state. assumes segmenttimes defined by RT.set_pipeline.
    """

    # define times to read
    if segment != -1:
        assert 'segmenttimes' in d, 'd must have segmenttimes defined'

        t0 = d['segmenttimes'][segment][0]
        t1 = d['segmenttimes'][segment][1]
        datetime = qa.time(qa.quantity((t1+t0)/2, 'd'),
                           form=['ymdhms'], prec=9)[0]
        logger.debug('Calculating uvw for segment %d' % (segment))
    else:
        datetime = 0

    (u, v, w) = calc_uvw(d['filename'], d['scan'],
                         datetime=datetime, bdfdir=d['bdfdir'])

    # cast to units of lambda at first channel.
    # -1 keeps consistent with ms reading convention
    u = u * d['freq_orig'][0] * (1e9/3e8) * (-1)
    v = v * d['freq_orig'][0] * (1e9/3e8) * (-1)
    w = w * d['freq_orig'][0] * (1e9/3e8) * (-1)

    return u.astype('float32'), v.astype('float32'), w.astype('float32')


def read_sources(sdmname):
    """ Use sdmpy to get all sources and ra,dec per scan as dict """

    sdm = getsdm(sdmname)
    sourcedict = {}

    for row in sdm['Field']:
        src = str(row.fieldName)
        sourcenum = int(row.sourceId)
        direction = str(row.referenceDir)
        (ra,dec) = [float(val) for val in direction.split(' ')[3:]]  # skip first two values in string

        sourcedict[sourcenum] = {}
        sourcedict[sourcenum]['source'] = src
        sourcedict[sourcenum]['ra'] = ra
        sourcedict[sourcenum]['dec'] = dec

    return sourcedict


def read_scans(sdmfile, bdfdir=''):
    """ Use sdmpy to get all scans and info needed for rtpipe as dict """

    sdm = getsdm(sdmfile, bdfdir)
    scandict = {}
    skippedscans = []

    for scan in sdm.scans():
        scannum = int(scan.idx)
        scandict[scannum] = {}

        intentstr = ' '.join(scan.intents)
        src = scan.source

        scandict[scannum]['source'] = src
        scandict[scannum]['intent'] = intentstr

        # bdf specific properties
        try:
            startmjd = scan.bdf.startTime
            nints = scan.bdf.numIntegration
            interval = scan.bdf.get_integration(0).interval
            endmjd = startmjd + (nints*interval)/(24*3600)
            bdfstr = scan.bdf.fname

            scandict[scannum]['startmjd'] = startmjd
            scandict[scannum]['endmjd'] = endmjd
            scandict[scannum]['duration'] = endmjd-startmjd
            scandict[scannum]['nints'] = nints
            scandict[scannum]['bdfstr'] = bdfstr

            # clear reference to nonexistent BDFs (either bad or not in standard locations)
            if (not os.path.exists(scandict[scannum]['bdfstr'])) or ('X1' in bdfstr):
                scandict[scannum]['bdfstr'] = None
                logger.debug('Invalid bdf for %d of %s' % (scannum, sdmfile) )

        except IOError:
            skippedscans.append(scannum)

    if skippedscans:
        logger.warn('No BDF found for scans {0}'.format(skippedscans))

    return scandict


def sdm2ms(sdmfile, msfile, scan, inttime='0'):
    """ Converts sdm to ms format for a single scan.
    msfile defines the name template for the ms. Should end in .ms,
    but "s<scan>" will be put in.
    scan is string of (sdm counted) scan number.
    inttime is string to feed to split command.
    gives option of integrated data down in time.
    """

    # fill ms file
    if os.path.exists(msfile):
        logger.debug('%s already set.' % msfile)
    else:
        logger.info('No %s found. Creating anew.' % msfile)
        if inttime != '0':
            logger.info('Filtering by int time.')
            call(['asdm2MS', '--ocm', 'co',
                             '--icm', 'co', '--lazy', '--scans',
                             scan, sdmfile, msfile + '.tmp'])
            cfg = tasklib.SplitConfig()  # configure split
            cfg.vis = msfile + '.tmp'
            cfg.out = msfile
            cfg.timebin = inttime
            cfg.col = 'data'
            cfg.antenna = '*&*'  # discard autos
            tasklib.split(cfg)  # run task
            # clean up
            rmtree(msfile+'.tmp')
        else:
            call(['asdm2MS', '--ocm', 'co',
                             '--icm', 'co', '--lazy', '--scans',
                             scan, sdmfile, msfile])

    return msfile


def getsdm(*args, **kwargs):
    """ Wrap sdmpy.SDM to get around schema change error """

    try:
        sdm = sdmpy.SDM(*args, **kwargs)
    except XMLSyntaxError:
        kwargs['use_xsd'] = False
        sdm = sdmpy.SDM(*args, **kwargs)

    return sdm
