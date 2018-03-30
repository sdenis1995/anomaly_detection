from functions import *
import numpy as np
import configparser

#for change point detection (using ecp package for R)
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
e = importr('ecp')
from rpy2.robjects import numpy2ri
numpy2ri.activate()

#loading config file
config = configparser.ConfigParser()
config.read('config.cfg')

try:
    import cPickle as pickle
except:
    import pickle

if not config['DEFAULT']['classifier_filename']:
    config['DEFAULT']['classifier_filename'] = './model.pkl'

if not config['DEFAULT']['ecp_min_cluster_size']:
    config['DEFAULT']['ecp_min_cluster_size'] = 15

if not config['DEFAULT']['ecp_significance_level']:
    config['DEFAULT']['ecp_significance_level'] = 0.05

with open(config['DEFAULT']['classifier_filename'], 'rb') as f:
    interval_model = pickle.load(f)

def get_interval_features(data):
    #dummy function, returns median and oscillation rate for every dynamic characteristic
    if len(data) == 0:
        raise Exception("Empty array is given")
    medians = np.median(data, axis=0)
    oscillation_rates = (np.max(data, axis=0)-np.min(data, axis=0))/np.average(data, axis=0)
    return np.append(medians, oscillation_rates)


def get_change_points(data):
    normalized_data = np.array(data)
    for l in range(len(normalized_data[0])):
        m = np.max(normalized_data[:, l])
        if m > 0.001:
            normalized_data[:, l] = normalized_data[:, l] / m

    estimated = e.e_divisive(normalized_data,
                             sig_lvl=float(config['DEFAULT']['ecp_significance_level']),
                             min_size=int(config['DEFAULT']['ecp_min_cluster_size']))
    estimated = np.array(estimated[estimated.names.index('estimates')], dtype=np.int64)
    estimated = estimated - 1
    estimated[-1] -= 1

    # this part is optional, we just needed to localize change point and get him its own interval,
    # so it wouldn't interfere with neighbouring intervals
    estimated = postfilter(normalized_data, estimated[1:-1])
    estimated.extend([0, len(data)])

    return np.unique(estimated)

def process_job(data, feature_selection_function = get_interval_features):
    if get_interval_features is None:
        raise(Exception("No interval feature selection function"))
    changepoints = get_change_points(data)
    interval_data = [None] * (len(changepoints) - 1)
    for i in range(len(changepoints) - 1):
        interval_data[i] = feature_selection_function(data[changepoints[i]:changepoints[i + 1]])
    interval_classes = interval_model.fit(interval_data)
    return changepoints, interval_classes
