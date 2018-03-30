import configparser
import time
from multiprocessing import Process, Queue
import numpy as np
from sklearn import model_selection
from random import shuffle
from sklearn.ensemble import RandomForestClassifier

config = configparser.ConfigParser()
config.read('config.cfg')

if not config['DEFAULT']['cross_val_cv']:
    config['DEFAULT']['cross_val_cv'] = 5

if not config['DEFAULT']['NumTreesForFeatureSelection']:
    config['DEFAULT']['NumTreesForFeatureSelection'] = 64

if not config['DEFAULT']['cross_val_iters_for_feature_addition']:
    config['DEFAULT']['cross_val_iters_for_feature_addition'] = 50

if not config['DEFAULT']['default_random_feature_count']:
    config['DEFAULT']['default_random_feature_count'] = 5

if not config['DEFAULT']['feature_addition_feature_limit']:
    config['DEFAULT']['feature_addition_feature_limit'] = 30

if not config['DEFAULT']['feature_addition_accuracy_limit']:
    config['DEFAULT']['feature_addition_accuracy_limit'] = 0.9


# Функция подсчета точностей при включении определенной характеристики (1 шаг)
def parallel_index_check_cv(features, indices_left, index_start, data, labels, queue, model, proc_count):
    for i in range(index_start, len(indices_left), proc_count):
        index = indices_left[i]
        feature_copy = list(features)
        feature_copy.append(index)
        tmp_data = np.array([[x[k] for k in feature_copy] for x in data])
        accuracies = []
        for j in range(int(config['DEFAULT'][''])):
            accuracies.extend(model_selection.cross_val_score(model, tmp_data, labels, cv=int(config['DEFAULT']['cross_val_cv'])))
        queue.put([np.average(accuracies), i])

# функция, реализуующая пошаговое включение до того, как будет включен хотя бы одна из характеристик из всех выделенных групп (определяются дальше)
def feature_addition_parallel(data, labels, base_features = None, n_procs = 8, breakpoint_type = 'limit', verbose = 0):
    if breakpoint_type != 'limit' or breakpoint_type != 'accuracy':
        raise(Exception("Wrong breakpoint type, acceptable values are 'limit' and 'accuracy'"))

    #the model we use is RandomForest, you can change the default parameters of it yourself to suit your problem
    model = RandomForestClassifier(n_estimators=int(config['DEFAULT']['NumTreesForFeatureSelection']))

    if base_features is None:
        feature_list = shuffle(list(range(len(data[0]))))[:int(config['DEFAULT']['default_random_feature_count'])]
    else:
        feature_list = list(base_features)

    indices_left = list(set(range(len(data[0]))) - set(feature_list))

    start_time = time.time()

    for i in range(len(indices_left)):
        # начало шага
        # считаем точности, полученные при включении определенной характеристики
        processes = []
        result_queue = Queue()
        for n_p in range(n_procs - 1):
            p = Process(target=parallel_index_check_cv, args=(feature_list, indices_left, n_p, data, labels, result_queue, model, n_procs))
            p.start()
            processes.append(p)
        parallel_index_check_cv(feature_list, indices_left, n_procs - 1, data, labels, result_queue, model, n_procs)
        accuracies = [None] * len(indices_left)
        for k in range(len(indices_left)):
            res = result_queue.get()
            accuracies[res[2]] = res[0]

        for p in processes:
            p.join()
        res_index = np.argmax(accuracies)
        if verbose == 2:
            print("{} index added".format(indices_left[res_index]))
        feature_list.append(indices_left[res_index])
        del indices_left[res_index]

        if verbose > 0:
            print("--- %s seconds passed ---" % (time.time() - start_time))

        if breakpoint_type == 'limit':
            if len(feature_list) >= int(config['DEFAULT']['feature_addition_feature_limit']):
                break
        if breakpoint_type == 'accuracy':
            if accuracies[res_index] > float(config['DEFAULT']['feature_addition_accuracy_limit']):
                break

    return feature_list
