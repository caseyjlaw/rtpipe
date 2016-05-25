import rtpipe.RT as rt
import rtpipe.parsecands as pc
import rtpipe.parsesdm as ps
import click, os, glob
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)
logger = logging.getLogger(__name__)


@click.group('rtpipe')
def cli():
    pass


@cli.command()
@click.argument('filename')
@click.option('--paramfile', default='')
@click.option('--bdfdir', default='')
@click.option('--scan', default=1)
def read(filename, paramfile, bdfdir, scan):
    """ Simple parse and return metadata for pipeline for first scan """

    filename = os.path.abspath(filename)

    scans = ps.read_scans(filename, bdfdir=bdfdir)
    logger.info('Scans, Target names:')
    logger.info('%s' % str([(ss, scans[ss]['source']) for ss in scans]))
    logger.info('Example pipeline:')
    state = rt.set_pipeline(filename, scan, paramfile=paramfile, logfile=False)


@cli.command()
@click.argument('filename', type=str)
@click.option('--scan', type=int, default=0)
@click.option('--paramfile', type=str, default='rtpipe_cbe.conf')
@click.option('--logfile', type=bool, default=False)
@click.option('--bdfdir', default='')
def searchone(filename, scan, paramfile, logfile, bdfdir):
    """ Searches one scan of filename

    filename is name of local sdm ('filename.GN' expected locally).
    scan is scan number to search. if none provided, script prints all.
    assumes filename is an sdm.
    """

    filename = os.path.abspath(filename)
    scans = ps.read_scans(filename, bdfdir=bdfdir)

    if scan != 0:
        d = rt.set_pipeline(filename, scan, paramfile=paramfile,
                            fileroot=os.path.basename(filename), logfile=logfile)
        rt.pipeline(d, range(d['nsegments']))

        # clean up and merge files
        pc.merge_segments(filename, scan)
        pc.merge_scans(os.path.dirname(filename), os.path.basename(filename), scans.keys())
    else:
        logger.info('Scans, Target names:')
        logger.info('%s' % str([(ss, scans[ss]['source']) for ss in scans]))
        logger.info('Example pipeline:')
        state = rt.set_pipeline(filename, scans.popitem()[0], paramfile=paramfile,
                                fileroot=os.path.basename(filename), logfile=logfile)


@cli.command()
@click.argument('filename')
@click.option('--snrmin', default=0.)
@click.option('--snrmax', default=999.)
@click.option('--bdfdir', default='')
def mergeall(filename, snrmin, snrmax, bdfdir):
    """ Merge cands/noise files over all scans

    Tries to find scans from filename, but will fall back to finding relevant files if it does not exist.
    """

    filename = os.path.abspath(filename)
    bignumber = 500

    if os.path.exists(filename):
        scans = ps.read_scans(filename, bdfdir=bdfdir)
        scanlist = sorted(scans.keys())
    else:
        logger.warn('Could not find file {0}. Estimating scans from available files.'.format(filename))
        filelist = glob.glob(os.path.join(os.path.dirname(filename), '*{0}_sc*pkl'.format(os.path.basename(filename))))
        try:
            scanlist = sorted(set([int(fn.rstrip('.pkl').split('_sc')[1].split('seg')[0]) for fn in filelist]))
        except IndexError:
            logger.warn('Could not parse filenames for scans. Looking over big range.')
            scanlist = range(bignumber)

    logger.info('Merging over scans {0}'.format(scanlist))

    for scan in scanlist:
        pc.merge_segments(filename, scan)
    pc.merge_scans(os.path.dirname(filename), os.path.basename(filename), scanlist, snrmin=snrmin, snrmax=snrmax)


@cli.command()
@click.argument('filename', type=str)
@click.option('--html', type=bool, default=True, help='Create html version')
@click.option('--basenb', type=str, default='', help='Full path to base notebook. Default to distribution version')
@click.option('--agdir', type=str, default='', help='Activegit repo for applying classifications')
def nbcompile(filename, html, basenb, agdir):
    """ Compile the baseinteract.ipynb notebook into an analysis notebook for filename """

    filename = os.path.abspath(filename)

    pc.nbcompile(os.path.dirname(filename), os.path.basename(filename), html=html, basenb=basenb, agdir=agdir)


if __name__ == '__main__':
    cli()
