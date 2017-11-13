# /usr/bin/env python3
import gc
import numpy as np
from skmultilearn.problem_transform import LabelPowerset
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import time
from sys import stderr

n_classes = 12289
n_labels = 9

coded_output_file_prefixes = ['pos_error_0o1', 'pos_error_0o4',
                              'pos_error_0o25', 'pos_perf']


def time_func(func):
    def wrapper(*args, **kwargs):
        print('Timing function: {}(...)'.format(func.__name__), file=stderr)
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print('{}(...) total time: {} seconds'.format(func.__name__,
                                                      end_time - start_time),
              file=stderr)
        return result

    return wrapper


@time_func
def process_output(output):
    processed_output = np.zeros(shape=(len(output), n_classes))
    for row in range(len(output)):
        for label in output[row]:
            processed_output[row, label] = 1
    return processed_output


@time_func
def load_data(file, output_process_func=process_output):
    print('Loading data from file:', file)
    data = np.loadtxt(file, dtype=int, delimiter=',', )
    print('Done.')

    data_in = data[:, :-n_labels]
    data_out = data[:, -n_labels:]

    data = None
    gc.collect()

    return data_in, output_process_func(data_out)


if __name__ == '__main__':

    results = dict()

    for prefix in coded_output_file_prefixes:
        training_file = 'coded_output/' \
                        '{}_new_training_data_step_2.0.txt'.format(prefix)
        test_file = 'coded_output/' \
                    '{}_new_test_data_step_2.0.txt'.format(prefix)

        train_in, train_out = load_data(training_file)
        test_in, test_out = load_data(test_file)

        classifier = LabelPowerset(LinearSVC())
        time_func(classifier.fit)(train_in, train_out)
        predictions = time_func(classifier.predict)(test_in)
        acc = accuracy_score(test_out, predictions)

        results[prefix] = acc

    print('''
    
--------- * ---------
    
Accuracy ratings per experiment:
    ''')

    for prefix, acc in results.items():
        print('{}: {}%'.format(prefix, acc * 100.0))
