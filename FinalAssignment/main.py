# /usr/bin/env python3
import gc
import numpy as np
from skmultilearn.problem_transform import LabelPowerset
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import time
from sys import stderr

n_classes = 12289
n_labels = 9

coded_output_file_prefixes = ['pos_error_0o1', 'pos_error_0o4',
                              'pos_error_0o25', 'pos_perf']


def time_func(func):
    def wrapper(*args, **kwargs):
        stderr.write('Timing function: {}(...)'.format(func.__name__))
        stderr.write('\n')
        stderr.flush()

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


@time_func
def main():
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
                                            cv=5,
                                            n_jobs=2)

        total_data_out = total_data_in = None
        gc.collect()

        # time_func(classifier.fit)(train_in, train_out)
        # predictions = time_func(classifier.predict)(test_in)
        # acc = accuracy_score(test_out, predictions)


        results[prefix] = scores.mean()

    print('''

    --------- * ---------

    Accuracy ratings per experiment (mean of 5-fold cross validation):
        ''')

    for prefix, acc in results.items():
        print('{}: {}%'.format(prefix, acc * 100.0))


if __name__ == '__main__':
    main()
