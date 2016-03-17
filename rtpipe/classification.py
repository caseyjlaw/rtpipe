import getopt, sys, os, math, re
from os.path import exists as pathexists
import numpy as np
from sklearn.externals import joblib # for loading classifier
from sklearn.preprocessing import Imputer
from rtpipe.parsecands import read_candidates
from pickle_utils import *
from sklearn_utils import *
from sh import git, mv

logging.basicConfig(level=logging.INFO)

## vla_classifier.py ##

# usage
#loc_stats, prop_stats = read_candidates(f_in)
#logging.info('Read %d candidates from %s.' % (len(loc_stats), f_in))
#prop_stats = np.array(prop_stats)
#feats = stat_features(prop_stats)
#scores = classify(feats, rbversion)

def load_classifier(clf_file):
    """ loads a pre-trained classifier from file """
    clf = joblib.load(clf_file)
    logging.info("loaded classifier from file %s"%clf_file)
    return clf


def classify(feats, rb_version, njobs=1, verbose=0):

    # validate RB version
    validate_rb_version(rb_version)
    f_clfpkl = get_rb_pkl(rb_version)

    # load classifier and update classifier parameters according to user input
    try:
        clf = load_classifier(f_clfpkl)
        clf.n_jobs  = njobs
        clf.verbose = verbose
        logging.info('generating predictions for %d samples...'% feats.shape[0])
        scores = clf.predict_proba(feats)[:,1]
    except:
        print "ERROR running the classifier"
        raise

    logging.info('classified predictions done.')

    return scores


## vla_versions.py ##

# usage
#(train_version, feat_version) = parse_and_validate_rbv(rb_version)
#print "TV: %s" % train_version
#print "FV: %s" % feat_version
#featnames = lookup_features(rb_version)
#(f_ids_reals, f_ids_bogus) = lookup_traindata(rb_version)

# define feature sets
feats_v1 = ['snr','loc1','loc2','specstd','specskew','speckur','imskew','imkur']

MAX_TRAIN_VERSION = 1
MAX_FEAT_VERSION = 1

def parse_rbv(rb_version):

    p = re.compile("tv(\d+)fv(\d+)")
    m = p.match(rb_version)
    if len(m.groups()) != 2:
            raise ValueError("Invalid rb_version passed: %d" % rb_version)
    return int(m.group(1)), int(m.group(2))

    
def validate_rb_version(rb_version):
    (train_version, feat_version) = parse_rbv(rb_version)
    if train_version < 1 or train_version > MAX_TRAIN_VERSION:
        raise ValueError("Invalid training version specified: %s" %  rb_version)
    if feat_version < 1 or feat_version > MAX_FEAT_VERSION:
        raise ValueError("Invalid feature version specified: %s" % rb_version)


def get_rb_pkl(rb_version, mode='old', repopath='.'):

    # load pickled classifier

    if mode == 'old':
        this_dir = os.path.split(os.path.abspath(__file__))[0]
        f_clfpkl = os.path.join(this_dir,"classify_ET.%s.pkl" % rb_version)
    elif mode == 'new':
        repo = git.bake(_cwd=repopath)
        repo.checkout(rb_version)
        f_clfpkl = os.path.join(repopath, 'classify_ET.pkl')  # fixed name?
    else:
        raise IOError("mode {0} not known".format(mode))

    if not os.path.isfile(f_clfpkl):
        raise IOError("classifier pkl file %s not found" % f_clfpkl)
    logging.info("f_clfpkl=%s" % f_clfpkl)

    return f_clfpkl


def update_rb_repo(f_clfpkl, rb_version, repopath='.', commitmsg=''):

    repo = git.bake(_cwd=repopath)
    mv(f_clfpkl, os.path.join(repopath, f_clfpkl))
    repo.add(f_clfpkl)
    repo.commit(m=commitmsg)
    repo.tag(rb_version)
    repo.push()


def lookup_features(rb_version):

    (train_version, feat_version) = parse_rbv(rb_version)

    if feat_version == 1:
        return feats_v1
    else:
        raise ValueError("RB Version does not exist: %s" % rb_version)


def lookup_traindata(rb_version):

    (train_version, feat_version) = parse_rbv(rb_version)

    if train_version == 1:
        return ('recovered_reals_20150225.pkl', 'bogus_20150225.pkl')
    else:
        raise ValueError("RB Version does not exist: %s" % rb_version)


def printUsage():
    print "USAGE: features.py version"


def get_master_feats():
    return master_feats


def extract_feats(data, feats_to_extract):

    indices = np.asarray([master_feats.index(feat) for feat in feats_to_extract])
    return data[:,indices]


## vla_utils.py ##


def stat_features(stats):
    return stats[:,[0,2,3,4,5,6,7,8]]
