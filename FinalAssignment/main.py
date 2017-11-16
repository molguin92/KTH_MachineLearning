# /usr/bin/env python3
import csv
import gc
import numpy as np
from scipy.sparse import dok_matrix
from sklearn.metrics import accuracy_score
from skmultilearn.problem_transform import LabelPowerset
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import time
from sys import stderr

n_classes = 12289
n_labels = 9

coded_output_file_prefixes = [
    'pos_error_0o1', 'pos_error_0o4',
    'pos_error_0o25', 'pos_perf']


def time_func(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        stderr.write('{}(...) total time: {} seconds'.format(func.__name__,
                                                             end_time -
                                                             start_time))
        stderr.write('\n')
        stderr.flush()
        return result

    return wrapper


@time_func
def process_output(output):
    processed_output = dok_matrix((len(output), n_labels * n_classes),
                                  dtype=np.uint8)

    for i, row in enumerate(output):
        for j, label in enumerate(row):
            processed_output[i, (j * n_classes) + label] = 1

    return processed_output.tocoo()


@time_func
def load_data(file, output_process_func=process_output):
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
    assert data_in.shape[0] == data_out.shape[0]
    shuffle_indices = np.random.permutation(data_in.shape[0])
    return data_in[shuffle_indices], data_out[shuffle_indices]


@time_func
def k_fold_cross_validation(cf, data_in, data_out, k=10):
    # shuffle data, then partition
    assert k > 1

    data_in, data_out = shuffle_data(data_in, data_out)
    data_in = np.split(data_in, k)
    data_out = np.split(data_out, k)

    validation_in = dok_matrix(data_in.pop(0))
    validation_out = dok_matrix(data_out.pop(0))

    results = []

    for d_in, d_out in zip(data_in, data_out):

        cf.fit(d_in, d_out)
        predictions = cf.predict(validation_in)
        errors = (predictions - validation_out).nnz
        accuracy = 1.0 - ((errors * 1.0) / predictions.size)

        results.append(accuracy)

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

        print('Custom cross-validation...')
        mean_k_fold, std_k_fold = k_fold_cross_validation(classifier,
                                                          total_data_in,
                                                          total_data_out,
                                                          k=k)

        print('Done.\nBuilt in cross-validation...')
        scores_built_in = time_func(cross_val_score)(classifier,
                                                     total_data_in,
                                                     total_data_out,
                                                     cv=k)

        print('Done')
        total_data_out = total_data_in = None
        gc.collect()

        results[prefix] = (scores_built_in.mean(), scores_built_in.std(),
                           mean_k_fold, std_k_fold)
        print('{}: {}'.format(prefix, results[prefix]))

    print('''

--------- * ---------

Accuracy ratings per experiment (mean of {}-fold cross validation):
        '''.format(k))

    for prefix, stats in results.items():
        print('{} -> built-in: {} (std: {}) | custom: {} (std: {})'.format(
            prefix, stats[0], stats[1], stats[2], stats[3]))


@time_func
def learn_and_predict():
    print('Building linear SVMs and predicting.')

    accuracies = dict()
    for prefix in coded_output_file_prefixes:
        training_file = 'coded_output/' \
                        '{}_new_training_data_step_2.0.txt'.format(prefix)
        test_file = 'coded_output/' \
                    '{}_new_test_data_step_2.0.txt'.format(prefix)

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
        print('{} accuracy: {}'.format(prefix, acc))
        test_out = None
        gc.collect()

        with open('results/{}_predictions.txt'.format(prefix), 'w') as f:
            writer = csv.writer(f)
            _pred = predictions.toarray()
            for sample in _pred:
                row = []
                for i, element in enumerate(sample):
                    if element == 1:
                        row.append(i % n_classes)
                writer.writerow(row)

        accuracies[prefix] = acc

    print('Accuracies:')
    for prefix, acc in accuracies.items():
        print(prefix, acc)


if __name__ == '__main__':
    cross_validation()
    learn_and_predict()
