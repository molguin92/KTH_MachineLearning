# /usr/bin/env python3
import numpy as np
from scipy import sparse
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC, OneClassSVM
import time

n_classes = 12289


def process_output_vector(output):
    print('Processing output vector...')
    output_coded = sparse.lil_matrix((len(output), n_classes), dtype=int)
    for i in range(len(output)):
        for label in output[i]:
            output_coded[i, label] = 1
    print('Done.')
    return output_coded


if __name__ == '__main__':

    start_time = time.time()

    print('Loading data from training file...')
    data = np.loadtxt('coded_output/pos_perf_new_training_data_step_2.0.txt',
                      dtype=int, delimiter=',', )
    print('Done.')
    print('Loading data from test file...')
    data_test = np.loadtxt('coded_output/pos_perf_new_test_data_step_2.0.txt',
                           dtype=int, delimiter=',', )
    print('Done.')

    # data = data[:30,:]

    output = data[:, -9:]
    input = sparse.csr_matrix(data[:, :-9])
    output_coded = process_output_vector(output)

    output_test = data_test[:, -9:]
    input_test = sparse.csr_matrix(data_test[:, :-9])
    output_coded_test = process_output_vector(output_test)

    print('Training classifier...')
    cf = BinaryRelevance(OneClassSVM())
    # cf = MLkNN(k=3)
    cf.fit(input, output_coded)
    print('Done.')

    print('Predicting:')
    predictions = cf.predict(input_test)

    correct = 0
    total = 0
    for i in range(predictions.shape[0]):
        if (predictions[i] == output_coded_test[i]):
            correct += 1

        total += 1

    print('Total:', total)
    print('Correct:', correct)
    print('Accuracy:', float(correct)/total)

    print("--- %s seconds ---" % (time.time() - start_time))
