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
            processed_output[i, (j + 1) * label] = 1

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
def cross_validation(k=10):
    print('Performing {}-fold validation on models.'.format(k))
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

        total_data_out = np.vstack((train_out, test_out))
        train_out = test_out = None
        gc.collect()

        classifier = LabelPowerset(LinearSVC())
        scores = time_func(cross_val_score)(classifier,
                                            total_data_in,
                                            total_data_out,
                                            cv=k)

        total_data_out = total_data_in = None
        gc.collect()

        results[prefix] = (scores.mean(), scores.std())

    print('''

--------- * ---------

Accuracy ratings per experiment (mean of {}-fold cross validation):
        '''.format(k))

    for prefix, stats in results.items():
        print('{}: {} (std: {})'.format(prefix, stats[0], stats[1]))


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
        test_out = None
        gc.collect()

        # todo: fix this:
        with open('results/{}_predictions.txt'.format(prefix), 'w') as f:
            writer = csv.writer(f)
            _pred = predictions.toarray()
            for sample in _pred:
                row = []
                for i, element in enumerate(sample):
                    if element == 1:
                        row.append(i)
                writer.writerow(row)

        accuracies[prefix] = acc

    print('Accuracies:')
    for prefix, acc in accuracies.items():
        print(prefix, acc)


if __name__ == '__main__':
    # learn_and_predict()
    cross_validation()
