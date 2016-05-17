import rtpipe.RT as rt
import rtpipe.parsecands as pc
import click, sdmreader, os
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

    sc, sr = sdmreader.read_metadata(filename, bdfdir=bdfdir)
    logger.info('Scans, Target names:')
    logger.info('%s' % str([(ss, sc[ss]['source']) for ss in sc]))
    logger.info('Example pipeline:')
    state = rt.set_pipeline(filename, scan, paramfile=paramfile, logfile=False)


@cli.command()
@click.argument('filename')
@click.option('--snrmin', default=0.)
@click.option('--snrmax', default=999.)
def mergeall(filename, snrmin=0, snrmax=999):
    """ Merge cands/noise files over all scans """

    filename = os.path.abspath(filename)

    sc, sr = sdmreader.read_metadata(filename)
    for scan in sc:
        pc.merge_segments(filename, scan)
    pc.merge_scans(os.path.dirname(filename), os.path.basename(filename), sc.keys(), snrmin=snrmin, snrmax=snrmax)


@cli.command()
@click.argument('filename', type=str)
@click.option('--scan', type=int, default=0)
@click.option('--paramfile', type=str, default='rtpipe_cbe.conf')
@click.option('--logfile', type=bool, default=False)
def searchone(filename, scan, paramfile, logfile):
    """ Searches one scan of filename

    filename is name of local sdm ('filename.GN' expected locally).
    scan is scan number to search. if none provided, script prints all.
    assumes filename is an sdm.
    """

    filename = os.path.abspath(filename)

    if scan != 0:
        d = rt.set_pipeline(filename, scan, paramfile=paramfile,
                            fileroot=os.path.basename(filename), logfile=logfile)
        rt.pipeline(d, range(d['nsegments']))

        # clean up and merge files
        pc.merge_segments(filename, scan)
        pc.merge_scans(os.path.dirname(filename), os.path.basename(filename), sc.keys())
    else:
        sc,sr = sdmreader.read_metadata(filename)
        print('Scans, Target names:')
        print('%s' % str([(ss, sc[ss]['source']) for ss in sc]))
        print('Example pipeline:')
        state = rt.set_pipeline(filename, sc.popitem()[0], paramfile=paramfile,
                                fileroot=os.path.basename(filename), logfile=logfile)


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
