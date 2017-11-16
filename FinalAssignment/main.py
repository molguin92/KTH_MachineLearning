# /usr/bin/env python3
import csv
import gc
import time
from sys import stderr

import numpy as np
from scipy.sparse import dok_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from skmultilearn.problem_transform import LabelPowerset
from multiprocessing import Pool

n_classes = 12289
n_labels = 9

coded_output_file_prefixes = [
    'pos_error_0o1', 'pos_error_0o4',
    'pos_error_0o25', 'pos_perf']


def time_func(func):
    """
    Decorator to time function execution time in seconds. Prints to stderr.
    :param func: Function to time.
    :return: Wrapper which times the function and prints the execution time
    to stderr.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        stderr.write('{}(...) total time: {} seconds\n'.format(func.__name__,
                                                               end_time -
                                                               start_time))
        stderr.flush()
        return result

    return wrapper


@time_func
def process_output(output):
    """
    Processes the given label set output matrix to the binary representation
    expected by scikit-multilearn
    :param output: Label set output matrix
    :return: Binary representation of input matrix
    """
    processed_output = dok_matrix((len(output), n_labels * n_classes),
                                  dtype=np.uint8)

    for i, row in enumerate(output):
        for j, label in enumerate(row):
            processed_output[i, (j * n_classes) + label] = 1

    return processed_output.tocoo()


@time_func
def load_data(file, output_process_func=process_output):
    """
    Loads a dataset from a correctly formatted input file.
    :param file: Path of file to read.
    :param output_process_func: Processing function to apply to the datasets
    output data.
    :return: A touple containing the input data and processed output data for
    the dataset.
    """
    print('Loading data from file:', file)
    data = np.loadtxt(file, dtype=np.uint16, delimiter=',', )
    print('Done.')

    data_in = data[:, :-n_labels]
    data_out = data[:, -n_labels:]

    data = None
    gc.collect()

    return data_in, output_process_func(data_out)


@time_func
def shuffle_data(data_in, data_out):
    """
    Row-wise shuffles the input and output data of a dataset while
    maintaining the congruency between their rows.
    :param data_in: Input data for the dataset.
    :param data_out: Output data for the dataset.
    :return: Shuffled datasets.
    """
    assert data_in.shape[0] == data_out.shape[0]
    shuffle_indices = np.random.permutation(data_in.shape[0])
    return data_in[shuffle_indices], data_out[shuffle_indices]


@time_func
def k_fold_cross_validation(cf, data_in, data_out, k=10):
    """
    Performs k_fold cross validation on a dataset given a specific classifier.
    :param cf: Classifier to use for the cross-validation
    :param data_in: Input data.
    :param data_out: Output data, labels.
    :param k: Optional k-fold parameter. Default is 10.
    :return: Tuple containing the average accuracy and standard deviation
    obtained through cross-validation.
    """
    assert k > 1

    # shuffle data, then partition
    data_in, data_out = shuffle_data(data_in, data_out)
    data_in = np.split(data_in, k)
    data_out = np.split(data_out, k)

    results = []

    # each split is used exactly once for validation
    for i in range(k):
        # train on i, validate on all others
        cf.fit(data_in[i], data_out[i])

        for j in range(k):
            if i == j:
                pass

            predictions = cf.predict(data_in[j])
            errors = (predictions - dok_matrix(data_out[j])).nnz
            results.append(1.0 - ((errors * 1.0) / predictions.size))

    results = np.array(results)
    return results.mean(), results.std()


@time_func
def cross_validation(k=10):
    print('Performing {}-fold cross validation on models.'.format(k))
    results = dict()

    for prefix in coded_output_file_prefixes:
        training_file = 'coded_output/' \
                        '{}_new_training_data_step_2.0.txt'.format(prefix)
        test_file = 'coded_output/' \
                    '{}_new_test_data_step_2.0.txt'.format(prefix)

        train_in, train_out = load_data(training_file)
        test_in, test_out = load_data(test_file)

        total_data_in = np.vstack((train_in, test_in))
        train_in = test_in = None
        gc.collect()

        total_data_out = np.vstack((train_out.toarray(),
                                    test_out.toarray()))
        train_out = test_out = None
        gc.collect()

        print('Beginning cross-validation for dataset {}...'.format(prefix))
        classifier = LabelPowerset(LinearSVC())

        # print('Custom cross-validation...')
        mean_k_fold, std_k_fold = k_fold_cross_validation(classifier,
                                                          total_data_in,
                                                          total_data_out,
                                                          k=k)

        # print('Done.\nBuilt in cross-validation...')
        # scores_built_in = time_func(cross_val_score)(classifier,
        #                                             total_data_in,
        #                                             total_data_out,
        #                                             cv=k)

        print('Done')
        total_data_out = total_data_in = None
        gc.collect()

        results[prefix] = (mean_k_fold, std_k_fold)
        # results[prefix] = (scores_built_in.mean(), scores_built_in.std(),
        #                   mean_k_fold, std_k_fold)
        # print('{}: {}'.format(prefix, results[prefix]))

    print('''

--------- * ---------

Accuracy ratings per experiment (mean of {}-fold cross validation):

        '''.format(k))

    for prefix, stats in results.items():
        print('{}: {} (std: {})'.format(prefix, stats[0], stats[1]))
        # for prefix, stats in results.items():
        #     print('{} -> built-in: {} (std: {}) | custom: {} (std: {
        # })'.format(
        #         prefix, stats[0], stats[1], stats[2], stats[3]))


@time_func
def train_test_svm(dataset_name):
    training_file = 'coded_output/' \
                    '{}_new_training_data_step_2.0.txt'.format(dataset_name)
    test_file = 'coded_output/' \
                '{}_new_test_data_step_2.0.txt'.format(dataset_name)

    train_in, train_out = load_data(training_file)
    test_in, test_out = load_data(test_file)

    classifier = LabelPowerset(LinearSVC())

    time_func(classifier.fit)(train_in, train_out)
    train_in = train_out = None
    gc.collect()

    predictions = time_func(classifier.predict)(test_in)
    test_in = None
    gc.collect()

    acc = accuracy_score(test_out, predictions)
    test_out = None
    gc.collect()

    return acc, predictions


@time_func
def parallel_learn_and_predict():
    print('Building linear SVMs and predicting.')

    with Pool(processes=3) as pool:
        results = pool.map(train_test_svm, coded_output_file_prefixes)

        for prefix, (acc, predictions) in zip(coded_output_file_prefixes,
                                              results):
            with open('results/{}_predictions.txt'.format(prefix), 'w') as f:
                writer = csv.writer(f)
                _pred = predictions.toarray()
                for sample in _pred:
                    row = []
                    for i, element in enumerate(sample):
                        if element == 1:
                            row.append(i % n_classes)
                    writer.writerow(row)

            print('{}: {}'.format(prefix, acc))

if __name__ == '__main__':
    # cross_validation()
    parallel_learn_and_predict()
