# /usr/bin/env python3
import csv
import gc
import time
from sys import stderr

import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
from scipy.sparse import vstack as sparse_vstack
from sklearn.svm import LinearSVC
from skmultilearn.problem_transform import LabelPowerset
from multiprocessing import Pool, Process, Event
import itertools
from psutil import cpu_percent, virtual_memory, cpu_count

n_classes = 12289
n_labels = 9

coded_output_file_prefixes = [
    'pos_error_0o1', 'pos_error_0o4',
    'pos_error_0o25', 'pos_perf']


class ProfilingProcess(Process):
    def __init__(self, ):
        Process.__init__(self)
        Process.name = 'LinearSVC_Profiling'
        self.exit = Event()
        self.cpu = list()
        self.ram = list()
        cpu_percent()

    def run(self):
        while not self.exit.is_set():
            cpu = cpu_percent(percpu=True)
            cpu.append(cpu_percent())
            self.cpu.append(cpu)

            memory = virtual_memory()
            self.ram.append(memory.total - memory.available)

            time.sleep(1)

        with open('results/profiling_data.txt', 'w') as f:
            header = ['cpu' + str(i) for i in range(cpu_count())]
            header.append('cpu_total')
            header.append('ram')

            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for cpu_sample, ram_sample in zip(self.cpu, self.ram):
                l_row = list(cpu_sample)
                l_row.append(ram_sample)
                dict = {header[i]: l_row[i] for i in range(len(header))}
                writer.writerow(dict)

    def terminate(self):
        self.exit.set()


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


def encode_output(output):
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

    return processed_output.tocsr()


def decode_output(encoded_output):
    """
    Decodes a binary indicator matrix into a 2-D vector of label vectors.
    Each row in the output corresponds to the label vector for a specific
    sample.
    :param encoded_output: Binary indicator matrix.
    :return: 2-D vector of label vectors.
    """
    n_zero = encoded_output.nonzero()
    decoded_output = np.empty(shape=(encoded_output.shape[0], n_labels),
                              dtype=np.uint16)
    for row, col in zip(*n_zero):
        value = col % n_classes
        decoded_col = int(col - value) / n_classes
        decoded_output[int(row), int(decoded_col)] = value

    return decoded_output


def load_data(file, output_process_func=encode_output):
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

    data_in = data[:, :-n_labels]
    data_out = data[:, -n_labels:]

    data = None
    gc.collect()

    return data_in, output_process_func(data_out)


@time_func
def load_dataset(dataset_name):
    training_file = 'coded_output/' \
                    '{}_new_training_data_step_2.0.txt'.format(dataset_name)
    test_file = 'coded_output/' \
                '{}_new_test_data_step_2.0.txt'.format(dataset_name)

    train_in, train_out = load_data(training_file)
    test_in, test_out = load_data(test_file)
    return train_in, train_out, test_in, test_out


def accuracy_score(x1, x2):
    assert x1.shape == x2.shape
    assert x1.nnz == x2.nnz
    errors = (x1 - x2).nnz / 2.0
    return 1.0 - (errors / x1.nnz)


def validate(cf, data_in, data_out, return_predictions=True):
    """
    Applies a classifier cf to the data and returns the accuracy score,
    and optionally the predictions.
    :param cf: Classifier
    :param data_in: Training input data.
    :param data_out: Training output data.
    :param get_predictions: Flag to indicate return predictions:
    :return: Accuracy score (double).
    """
    predictions = cf.predict(data_in)
    acc = accuracy_score(data_out, predictions)
    if not return_predictions:
        return acc
    else:
        return acc, predictions


def shuffle_data(data_in, data_out):
    """
    Row-wise shuffles the input and output data of a dataset while
    maintaining the congruency between their rows.
    :param data_in: Input data for the dataset.
    :param data_out: Output data for the dataset.
    :return: Shuffled datasets.
    """
    try:
        assert data_in.shape[0] == data_out.shape[0]
    except AssertionError:
        print(data_in.shape[0], data_out.shape[0])
        raise

    shuffle_indices = np.random.permutation(data_in.shape[0])
    return data_in[shuffle_indices], data_out[shuffle_indices]


def cross_validation_fold(index, splits_in, splits_out):
    """
    k-fold cross-validation "fold": performs validation using exactly
    one of the splits as validation set and the rest of the dataset
    as training data.
    :param index: Index of the split to use as validation data
    :param splits_in: List of splits of the original dataset inputs
    :param splits_out: List of splits of the origina dataset outputs
    :return: The accuracy score for a LinearSVC trained on all the
    splits except <index> and then validated on split <index>
    """
    validation_in = splits_in[index]
    validation_out = splits_out[index]
    cf = LabelPowerset(LinearSVC())

    # train on all splits except split <index>
    cf.fit(np.vstack(splits_in[:index] + splits_in[index + 1:]),
           sparse_vstack(splits_out[:index] + splits_out[index + 1:]))

    # validate on split <index>
    return validate(cf, validation_in, validation_out, return_predictions=False)


@time_func
def k_fold_cross_validation(dataset, k=10):
    """
    Performs k_fold cross validation on a dataset using LinearSVCs
    :param dataset: Dataset tuple in format (train_in, train_out, test_in,
    test_out)
    :param k: Optional k-fold parameter. Default is 10.
    :return: Tuple containing the average accuracy and standard deviation
    obtained through cross-validation.
    """

    assert k > 1

    # label data has to be dense to be shuffled
    # in the same order as the input data
    data_in = np.vstack((dataset[0], dataset[2]))
    data_out = np.vstack((dataset[1].toarray(), dataset[3].toarray()))

    # shuffle data, then partition
    data_in, data_out = shuffle_data(data_in, data_out)
    data_in = np.split(data_in, k)
    data_out = np.split(data_out, k)

    # re-sparsify?
    data_out = list(map(csr_matrix, data_out))

    results = None
    with Pool(processes=4) as pool:
        results = pool.starmap(cross_validation_fold,
                               zip(range(k),
                                   itertools.repeat(data_in),
                                   itertools.repeat(data_out)))

    results = np.array(results)
    return results.mean(), results.std()


@time_func
def cross_validate(datasets):
    print('Cross-validating model...')
    results = list(map(lambda dset: k_fold_cross_validation(dset), datasets))

    for name, stats in zip(coded_output_file_prefixes, results):
        print('Cross-validation {}: {} (std: {})'.format(name,
                                                         stats[0],
                                                         stats[1]))


def train_test_svm(dataset, return_predictions=True):
    train_in, train_out, test_in, test_out = dataset

    classifier = LabelPowerset(LinearSVC())
    time_func(classifier.fit)(train_in, train_out)

    acc, predictions = time_func(validate)(classifier, test_in,
                                           test_out, return_predictions)

    if return_predictions:
        return acc, predictions
    else:
        return acc


@time_func
def parallel_learn_and_predict(datasets):
    # datasets: array of tuples (train_in, train_out, test_in, test_out)
    print('Building linear SVMs and predicting...')
    with Pool(processes=4) as pool:
        results = pool.map(train_test_svm, datasets)

        predictions = []
        accs = []
        for acc, pred in results:
            accs.append(acc)
            predictions.append(pred)

        decoded_outputs = pool.map(decode_output, predictions)

        for prefix, acc in zip(coded_output_file_prefixes, accs):
            print('{}: {}'.format(prefix, acc))

        print('Writing predictions to files...')
        for prefix, output in zip(coded_output_file_prefixes, decoded_outputs):
            with open('results/{}_predictions.txt'.format(prefix), 'w') as f:
                writer = csv.writer(f)
                writer.writerows(output)


if __name__ == '__main__':
    profiling_process = ProfilingProcess()
    profiling_process.start()
    time.sleep(2)

    datasets = list(map(load_dataset, coded_output_file_prefixes))
    parallel_learn_and_predict(datasets)
    print('\n---- * ----\n')
    cross_validate(datasets)

    gc.collect()

    time.sleep(2)
    profiling_process.terminate()
    profiling_process.join()
