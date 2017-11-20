import numpy as np
from sklearn.cluster import KMeans
from math import sqrt, inf
from pandas import read_csv
from sys import argv
# import seaborn as sns
# from matplotlib import pyplot as plt

rbf_count = 25 # int(argv[1])
alpha = 0.25 # float(argv[2])
lambda2 = 0.3 # float(argv[3])
epoch_count = 1000000 # int(argv[4])
minibatch_factor = 70 # int(argv[5])

syn_input_data = read_csv('data/input.csv',delimiter=',').as_matrix()
letor_input_data = np.genfromtxt('data/Querylevelnorm_X.csv', delimiter=',')
syn_output_data = read_csv('data/output.csv', delimiter=',').as_matrix()
syn_output_data = syn_output_data.reshape([-1, 1])
letor_output_data = np.genfromtxt('data/Querylevelnorm_t.csv',
                                  delimiter=',').reshape([-1, 1])


ratio = {'training': 0.8, 'validation': 0.9, 'test': 1}
# rbf_count = 30

def preprocess(input_data, output_data):
    row_count, _ = input_data.shape

    train_count = int(round(ratio['training'] * row_count))
    val_count = int(round(ratio['validation'] * row_count))
    test_count = int(round(ratio['test'] * row_count))

    train_labels = output_data[:train_count, :]
    val_labels = output_data[train_count:val_count, :]
    test_labels = letor_output_data[val_count:, :]

    train_vectors = input_data[:train_count:1, :]
    val_vectors = input_data[train_count:val_count:1, :]
    test_vectors = input_data[val_count:test_count:1, :]

    return train_labels, val_labels, test_labels,\
        train_vectors, val_vectors, test_vectors


def compute_design_matrix(X, centers, spreads):
    # use broadcast
    basis_func_outputs = np.exp(np.sum(np.matmul(
        X - centers, spreads) * (X - centers), axis=2) / (-2)).T
    # insert ones to the 1st col
    return np.insert(basis_func_outputs, 0, 1, axis=1)


def closed_form_sol(L2_lambda, design_matrix, output_data):
    return np.linalg.solve(L2_lambda * np.identity(design_matrix.shape[1]) +
                           np.matmul(design_matrix.T, design_matrix),
                           np.matmul(design_matrix.T, output_data)).flatten()


def SGD_sol(learning_rate, minibatch_size, num_epochs,
            L2_lambda, design_matrix, output_data):
    N, _ = design_matrix.shape
    # You can try different mini-batch size size
    # Using minibatch_size = N is equivalent to standard gradient descent
    # Using minibatch_size = 1 is equivalent to stochastic gradient descent
    # In this case, minibatch_size = N is better
    weights = np.zeros([1, rbf_count + 1])
    # The more epochs the higher training accuracy. When set to 1000000,
    # weights will be very close to closed_form_weights.
    # But this is unnecessary
    prev = current = inf
    run_flag = True
    for epoch in range(num_epochs):
        if(run_flag):
            for i in range(int(round(N / minibatch_size))):
                if(current <= prev):
                    prev = current
                    lower_bound = i * minibatch_size
                    upper_bound = min((i + 1) * minibatch_size, N)
                    Phi = design_matrix[lower_bound: upper_bound, :]
                    t = output_data[lower_bound: upper_bound, :]
                    E_D = np.matmul((np.matmul(Phi, weights.T) - t).T, Phi)
                    E = (E_D + L2_lambda * weights) / minibatch_size
                    weights = weights - learning_rate * E
                else:
                    run_flag = False
                    break
            current = np.linalg.norm(E)
        else:
            break
    return weights.flatten()


def matrix_calc(vector, M):
    N, D = vector.shape
    centers = KMeans(n_clusters=M).fit(vector)
    spreads = np.zeros((M, D, D))
    holder = np.zeros((D, D, M))
    for i in range(M):
        x = vector[np.where(centers.labels_ == i)]
        temp = np.matmul(np.cov(x.T), np.identity(D))
        holder[:, :, i] = temp
    spreads = np.reshape(holder, (M, D, D))
    centroids = centers.cluster_centers_
    centroids = centroids[:, np.newaxis, :]
    X = vector[np.newaxis, :, :]

    return compute_design_matrix(X, centroids, spreads)


def erms(val_labels, predicted_labels, N):
    error = val_labels - predicted_labels
    error = error**2
    erms = sqrt(((sum(error)) / N))
    return erms


def evaluate(input_data, output_data):
    train_labels, val_labels, test_labels, train_vectors, val_vectors,\
        test_vectors = preprocess(input_data, output_data)

    train_matrix = matrix_calc(train_vectors, M=rbf_count)
    validation_matrix = matrix_calc(test_vectors, M=rbf_count)

    closed_form_weights = closed_form_sol(L2_lambda=lambda2,
                                          design_matrix=train_matrix,
                                          output_data=train_labels)[
        :, np.newaxis]

    sgd_weights = SGD_sol(learning_rate=alpha,
                          minibatch_size=int(round(
                              train_vectors.shape[0] / minibatch_factor)),
                          num_epochs=epoch_count,
                          L2_lambda=lambda2,
                          design_matrix=train_matrix,
                          output_data=train_labels)[:, np.newaxis]

    val_test_cf = np.matmul(validation_matrix, closed_form_weights)
    val_test_sgd = np.matmul(validation_matrix, sgd_weights)

    print(erms(val_labels, val_test_cf, val_vectors.shape[0]))
    print(erms(val_labels, val_test_sgd, val_vectors.shape[0]))

evaluate(letor_input_data, letor_output_data)
evaluate(syn_input_data, syn_output_data)
