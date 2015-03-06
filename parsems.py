import casautil
import os, pickle, time, string
import numpy as n

# CASA initialization
ms = casautil.tools.ms()
tb = casautil.tools.table()
qa = casautil.tools.quanta()

def get_metadata(filename, scan=0, datacol='data', spw=[], chans=[], selectpol=[]):
    """ Function to scan data (a small read) and define parameters used elsewhere.
    filename needs full path.
    Examples include, read/bgsub windows, image grid, memory profile.
    If pickle file doesn't exist, it creates one. 
    Either way, dictionary is returned with info.
    """

    # create primary state dictionary
    d = {}

    # define working area (probably should be done elsewhere...)
    ss = filename.rstrip('/').split('/')
    if len(ss) > 1:
        d['filename'] = ss[-1]
        d['workdir'] = string.join(ss[:-1], '/') + '/'
        os.chdir(d['workdir'])
    else:
        d['filename'] = filename
        d['workdir'] = os.getcwd() + '/'

    # set misc params
    d['datacol'] = datacol

    # read metadata either from pickle or ms file
    pklname = d['filename'].rstrip('.ms') + '_init2.pkl'
    if os.path.exists(pklname):
        print 'Initializing pickle found at %s.' % pklname
        pkl = open(pklname, 'r')
        try:
            stuff = pickle.load(pkl)
        except EOFError:
            print 'Bad pickle file. Exiting...'
            return 1

        # load stuff into d
        for key in stuff.keys():
            d[key] = stuff[key]

    else:
        print 'No initialization pickle found. Making anew...'

        # find polarizations in MS
        tb.open(d['filename']+'/POLARIZATION')
        d['pols_orig'] = tb.getcol('CORR_TYPE').flatten()
        d['npol_orig'] = len(d['pols_orig'])
        tb.close()

        print 'Opening %s...' % d['filename']
        # find spectral and scan info
        ms.open(d['filename'])
        md = ms.metadata()

        # define ants and baselines
        d['nants'] = md.nantennas()
        d['ants'] = [int(md.antennanames(aa)[0][2:]) for aa in range(d['nants'])]
        d['nbl'] = md.nbaselines()
        d['blarr'] = n.array([[d['ants'][i],d['ants'][j]] for i in range(d['nants'])  for j in range(i+1, d['nants'])])
        # find spw info
        d['scans'] = md.scannumbers()

        # find scan info
        d['scansummary'] = ms.getscansummary()
        d['scanlist'] = sorted(d['scansummary'].keys())

        # find data structure
        print 'Reading a little data from each scan...'
        ms.selectinit(datadescid=0)  # reset select params for later data selection
        nints_snip = 10
        orig_spws_all = {}; freq_orig_all = {}
        urange = {}; vrange = {}
        for ss in d['scanlist']:
            # compile spws per scan
            orig_spws0 = md.spwsforscan(int(ss))
            orig_spws_all[ss] = [sorted(zip(orig_spws0, [md.chanfreqs(spw0)[0]/1e9 for spw0 in orig_spws0]), key=lambda ss:ss[1])[i][0] for i in range(len(orig_spws0))]
            ff = n.array([])
            for spw0 in orig_spws0:
                ff = n.concatenate( (ff, md.chanfreqs(spw0)/1e9) ).astype('float32')   # save freq in GHzx
            freq_orig_all[ss] = ff

            # compile times per scan
            starttime_mjd0 = d['scansummary'][ss]['0']['BeginTime']
            inttime0 = d['scansummary'][ss]['0']['IntegrationTime']
            starttime = qa.getvalue(qa.convert(qa.time(qa.quantity(starttime_mjd0,'d'),form=['ymd'], prec=9)[0], 's'))[0]
            stoptime = qa.getvalue(qa.convert(qa.time(qa.quantity(starttime_mjd0+(nints_snip*inttime0)/(24.*60*60), 'd'), form=['ymd'], prec=9)[0], 's'))[0]  # nints+1 to be avoid buffer running out and stalling iteration
            selection = {'time': [starttime, stoptime], 'uvdist': [1., 1e10]}    # exclude auto-corrs
            ms.select(items = selection)
            ms.iterinit(['TIME'], 10*inttime0)  # read 10 ints
            ms.iterorigin()
            da = ms.getdata(['axis_info', 'u', 'v', 'w'])
            urange[ss] = da['u'].max() - da['u'].min()
            vrange[ss] = da['v'].max() - da['v'].min()
        d['urange'] = urange
        d['vrange'] = vrange
        d['freq_orig_all'] = freq_orig_all
        d['orig_spws_all'] = orig_spws_all

        ms.close()
        # save initialization to pickle
        pkl = open(pklname, 'wb')
        pickle.dump((d), pkl)
        pkl.close()

    ### Now extract refine info for given scan, chans, etc. ###

    # refine pols info
    pols = []
    for pol in d['pols_orig']:
        if len(selectpol):  # if lesser set is requested, filter here
            if casautil.pol_names[pol] in selectpol:
                pols.append(casautil.pol_names[pol])
        else:
            pols.append(casautil.pol_names[pol])
    d['pols'] = pols
    d['npol'] = len(pols)

    # refine spw/freq info
    d['scan'] = d['scanlist'][scan]
    d['orig_spws'] = n.array(d['orig_spws_all'][d['scan']])
    if len(spw):
        d['spwlist'] = d['orig_spws'][spw]
    else:
        d['spwlist'] = d['orig_spws']

    d['spw'] = sorted(d['spwlist'])
    allfreq = d['freq_orig_all'][d['scan']]
    chperspw = len(allfreq)/len(d['orig_spws'])
    spwch = []
    for ss in d['orig_spws']:
        if ss in d['spwlist']:
            spwch.extend(range(ss*chperspw, (ss+1)*chperspw))
    d['freq_orig'] = allfreq[spwch]
           
    d['nspw'] = len(d['spwlist'])
    if len(chans):
        d['freq'] = d['freq_orig'][chans]
        d['chans'] = chans
    else:
        d['freq'] = d['freq_orig']
        d['chans'] = range(len(d['freq']))
    d['nchan'] = len(d['freq'])

    # define integrations for given scan
    d['nints'] = d['scansummary'][d['scan']]['0']['nRow']/(d['nbl']*d['npol'])
    inttime0 = d['scansummary'][d['scan']]['0']['IntegrationTime'] # estimate of inttime from first scan

    # define ideal res/npix
    d['uvres_full'] = n.round(25./(3e-1/d['freq'].max())/2).astype('int')    # full VLA field of view. assumes freq in GHz
    # **this may let vis slip out of bounds. should really define grid out to 2*max(abs(u)) and 2*max(abs(v)). in practice, very few are lost.**
    urange = d['urange'][d['scan']]*d['freq'].max() * (1e9/3e8)
    vrange = d['vrange'][d['scan']]*d['freq'].max() * (1e9/3e8)
    powers = n.fromfunction(lambda i,j: 2**i*3**j, (14,10), dtype='int')   # power array for 2**i * 3**j
    rangex = n.round(urange).astype('int')
    rangey = n.round(vrange).astype('int')
    largerx = n.where(powers-rangex/d['uvres_full'] > 0, powers, powers[-1,-1])
    p2x, p3x = n.where(largerx == largerx.min())
    largery = n.where(powers-rangey/d['uvres_full'] > 0, powers, powers[-1,-1])
    p2y, p3y = n.where(largery == largery.min())
    d['npixx_full'] = (2**p2x * 3**p3x)[0]
    d['npixy_full'] = (2**p2y * 3**p3y)[0]

    # define times
    d['starttime_mjd'] = d['scansummary'][d['scan']]['0']['BeginTime']
    d['inttime'] = d['scansummary'][d['scan']]['0']['IntegrationTime']

    # summarize metadata
    print 'Metadata summary:'
    print '\t Using scan %d (index %d)' % (int(d['scan']), scan)
    print '\t nants, nbl: %d, %d' % (d['nants'], d['nbl'])
    print '\t mid-freq, nspw, nchan: %.3f, %d, %d' % (d['freq'].mean(), d['nspw'], d['nchan'])
    print '\t inttime: %.3f s' % (d['inttime'])
    print '\t %d polarizations: %s' % (d['npol'], d['pols'])
    print '\t Ideal uvgrid npix=(%d,%d) and res=%d' % (d['npixx_full'], d['npixy_full'], d['uvres_full'])

    return d

def readiterinit(d):
    """ Prepare to read data with ms.iter*
    """

    # set requested time range based on given parameters
    starttime_mjd = d['starttime_mjd']
    timeskip = d['inttime']*d['nskip']
    starttime = qa.getvalue(qa.convert(qa.time(qa.quantity(starttime_mjd+timeskip/(24.*60*60),'d'),form=['ymd'], prec=9)[0], 's'))[0]
    stoptime = qa.getvalue(qa.convert(qa.time(qa.quantity(starttime_mjd+(timeskip+(d['nints']+1)*d['inttime'])/(24.*60*60), 'd'), form=['ymd'], prec=9)[0], 's'))[0]  # nints+1 to be avoid buffer running out and stalling iteration
    print 'Time of first integration:', qa.time(qa.quantity(starttime_mjd,'d'),form=['ymd'],prec=9)[0]
    print 'Reading times %s to %s in %d iterations' % (qa.time(qa.quantity(starttime_mjd+timeskip/(24.*60*60),'d'),form=['hms'], prec=9)[0], qa.time(qa.quantity(starttime_mjd+(timeskip+(d['nints']+1)*d['inttime'])/(24.*60*60), 'd'), form=['hms'], prec=9)[0], d['nthread'])

    # read data into data structure
    ms.open(d['filename'])
    if len(d['spwlist']) == 1:
        ms.selectinit(datadescid=d['spwlist'][0])
    else:
        ms.selectinit(datadescid=0, reset=True)    # reset includes spw in iteration over time
    selection = {'time': [starttime, stoptime], 'uvdist': [1., 1e10], 'antenna1': d['ants'], 'antenna2': d['ants']}    # exclude auto-corrs
    ms.select(items = selection)
    ms.selectpolarization(d['pols'])
    ms.iterinit(['TIME'], 0, d['iterint']*d['nbl']*d['nspw']*d['npol'], adddefaultsortcolumns=False)
    iterstatus = ms.iterorigin()

def readiter(d):
    """ Read iteration of size iterint
    """

    da = ms.getdata([d['datacol'],'axis_info','u','v','w','flag','data_desc_id'], ifraxis=True)
    good = n.where((da['data_desc_id']) == d['spwlist'][0])[0]   # take first spw                                                
    time0 = da['axis_info']['time_axis']['MJDseconds'][good]
    data0 = n.transpose(da[d['datacol']], axes=[3,2,1,0])[good]
    flag0 = n.transpose(da['flag'], axes=[3,2,1,0])[good]
    u0 = da['u'].transpose()[good] * d['freq_orig'][0] * (1e9/3e8)    # uvw are in m, so divide by wavelength of first chan to set in lambda
    v0 = da['v'].transpose()[good] * d['freq_orig'][0] * (1e9/3e8)
    w0 = da['w'].transpose()[good] * d['freq_orig'][0] * (1e9/3e8)

    if len(d['spwlist']) > 1:
        for spw in d['spwlist'][1:]:
            good = n.where((da['data_desc_id']) == spw)[0]
            data1 = n.transpose(da[d['datacol']], axes=[3,2,1,0])[good]
            data0 = n.concatenate( (data0, data1), axis=2 )
            flag0 = n.concatenate( (flag0, n.transpose(da['flag'], axes=[3,2,1,0])[good]), axis=2 )

    del da
    data0 = data0[:,:,d['chans'],:] * n.invert(flag0[:,:,d['chans'],:])   # flag==1 means bad data (for vla)                     
    iterstatus = ms.iternext() 

    return data0.astype('complex64'), u0.astype('float32'), v0.astype('float32'), w0.astype('float32'), time0.astype('float32')

def readsegment(d, segment):
    """ Prepare to read segment of data
    """

    d['segment'] = segment
    # set requested time range based on given parameters
    starttime = qa.getvalue(qa.convert(qa.time(qa.quantity(d['segmenttimes'][segment,0],'d'),form=['ymd'], prec=9)[0], 's'))[0]
    stoptime = qa.getvalue(qa.convert(qa.time(qa.quantity(d['segmenttimes'][segment,1],'d'),form=['ymd'], prec=9)[0], 's'))[0]
    print 'Reading segment %d/%d, times %s to %s' % (segment, len(d['segmenttimes'])-1, qa.time(qa.quantity(starttime/(24*3600),'d'),form=['hms'], prec=9)[0], qa.time(qa.quantity(stoptime/(24*3600), 'd'), form=['hms'], prec=9)[0])

    # read data into data structure
    ms.open(d['workdir'] + d['filename'])
    if len(d['spwlist']) == 1:
        ms.selectinit(datadescid=d['spwlist'][0])
    else:
        ms.selectinit(datadescid=0, reset=True)    # reset includes spw in iteration over time
    selection = {'time': [starttime, stoptime], 'uvdist': [1., 1e10]}
#    selection = {'time': [starttime, stoptime], 'uvdist': [1., 1e10], 'antenna1': d['ants'], 'antenna2': d['ants']}    # **this misses ants for some reason!**
    ms.select(items = selection)
    ms.selectpolarization(d['pols'])
    da = ms.getdata([d['datacol'],'axis_info','u','v','w','flag','data_desc_id'], ifraxis=True)
    good = n.where((da['data_desc_id']) == d['spwlist'][0])[0]   # take first spw                                                
    time0 = da['axis_info']['time_axis']['MJDseconds'][good]
    data0 = n.transpose(da[d['datacol']], axes=[3,2,1,0])[good]
    flag0 = n.transpose(da['flag'], axes=[3,2,1,0])[good]
    u0 = da['u'].transpose()[good] * d['freq_orig'][0] * (1e9/3e8)    # uvw are in m, so divide by wavelength of first chan to set in lambda
    v0 = da['v'].transpose()[good] * d['freq_orig'][0] * (1e9/3e8)
    w0 = da['w'].transpose()[good] * d['freq_orig'][0] * (1e9/3e8)

    if len(d['spwlist']) > 1:
        for spw in d['spwlist'][1:]:
            good = n.where((da['data_desc_id']) == spw)[0]
            data1 = n.transpose(da[d['datacol']], axes=[3,2,1,0])[good]
            data0 = n.concatenate( (data0, data1), axis=2 )
            flag0 = n.concatenate( (flag0, n.transpose(da['flag'], axes=[3,2,1,0])[good]), axis=2 )

    del da
    data0 = data0[:,:,d['chans'],:] * n.invert(flag0[:,:,d['chans'],:])   # flag==1 means bad data (for vla)                     

    return data0.astype('complex64'), u0.astype('float32'), v0.astype('float32'), w0.astype('float32')
