from functions import *
import numpy as np
import configparser
from sklearn.ensemble import RandomForestClassifier

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

if not config['DEFAULT']['NumTreesForClassifier']:
    config['DEFAULT']['NumTreesForClassifier'] = 256

if not config['DEFAULT']['ecp_significance_level']:
    config['DEFAULT']['ecp_significance_level'] = 0.05



def get_interval_features(data):
    # dummy function, returns median and oscillation rate for every dynamic characteristic
    if len(data) == 0:
        raise Exception("Empty array is given")
    medians = np.median(data, axis=0)
    oscillation_rates = (np.max(data, axis=0)-np.min(data, axis=0))/np.average(data, axis=0)
    return np.append(medians, oscillation_rates)

#dummy function for determining the class of the job based on interval classes and amount of processors that job was run on
def get_job_class(timestamps, changepoints, interval_classes, proc_count):
    result_class = 1
    distribution = np.zeros((3))
    for i in range(len(interval_classes)):
        distribution[interval_classes[i] - 1] += (timestamps[changepoints[i + 1]] - timestamps[changepoints[i]])
    # to convert into hours
    distribution /=  3600

    # to detect if suspicious
    if (distribution[2] * proc_count > 100) or \
        (distribution[2] > 1) or \
        (distribution[2] >= distribution[1] and distribution[2] >= distribution[0]):
        result_class = 3

    #to detect if abnormal
    if (distribution[1] *proc_count > 100) or \
        (distribution[1] > 1) or \
        (distribution[1] >= distribution[0] and distribution[1] >= distribution[2]):
        result_class = 2

    return result_class

class AnomalyDetector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=int(config['DEFAULT']['NumTreesForClassifier']))

    # method for fitting random forest classifier (interval classification)
    def fit(self, data, labels):
        # data : array of elements representing features of an interval
        # labels : array of classes for the intervals
        self.model.fit(data, labels)

    # load RandomForest model from file (pickled)
    def load_model(self, filename=None):
        if filename is None:
            filename = config['DEFAULT']['classifier_filename']
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)

    # save RandomForest model in file (using pickle)
    def save_model(self, filename=None):
        if filename is None:
            filename = config['DEFAULT']['classifier_filename']
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    # get change point locations in the data
    def get_change_points(self, data):
        # ecp relies on normalization, otherwise its decision can depend on only one or two features
        normalized_data = np.array(data)
        for l in range(len(normalized_data[0])):
            m = np.max(normalized_data[:, l])
            if m > 0.001:
                normalized_data[:, l] = normalized_data[:, l] / m

        # using ecp package to get change points of this multivariate time series
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

    # get job change points and interval classes for overall job classification based on these classes
    def process_job(self, data, feature_selection_function=get_interval_features):
        if get_interval_features is None:
            raise(Exception("No interval feature selection function"))
        changepoints = self.get_change_points(data)
        interval_data = [None] * (len(changepoints) - 1)
        for i in range(len(changepoints) - 1):
            interval_data[i] = feature_selection_function(data[changepoints[i]:changepoints[i + 1]])
        interval_classes = self.model.predict(interval_data)
        return changepoints, interval_classes

    def predict(self, data, proc_count, timestamps, feature_selection_function = get_interval_features):
        if get_interval_features is None:
            raise (Exception("No interval feature selection function"))
        changepoints, interval_classes = self.process_job(data, feature_selection_function)
        return get_job_class(timestamps, changepoints, interval_classes, proc_count)