# /usr/bin/env python3
import gc
import numpy as np
from scipy import sparse
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import LabelPowerset
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import time
from sys import stderr

n_classes = 12289
n_labels = 9


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
def load_data():
    print('Loading data from training file...')
    data = np.loadtxt('coded_output/pos_perf_new_training_data_step_2.0.txt',
                      dtype=int, delimiter=',', )
    print('Done.')

    train_in = data[:, :-n_labels]
    train_out = data[:, -n_labels:]

    data = None
    gc.collect()

    print('Loading data from test file...')
    data = np.loadtxt('coded_output/pos_perf_new_test_data_step_2.0.txt',
                      dtype=int, delimiter=',', )
    print('Done.')

    test_in = data[:, :-n_labels]
    test_out = data[:, -n_labels:]

    data = None
    gc.collect()

    return train_in, process_output(train_out), \
           test_in, process_output(test_out)


if __name__ == '__main__':
    train_in, train_out, test_in, test_out = load_data()
    classifier = LabelPowerset(LinearSVC())
    time_func(classifier.fit)(train_in, train_out)
    predictions = time_func(classifier.predict)(test_in)

    print('Accuracy: {}%'.format(100.0 * accuracy_score(test_out, predictions)))
